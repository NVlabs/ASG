# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# [x] train resnet101 with proxy guidance on visda17
# [x] evaluation on visda17

import argparse
import os
import sys
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pdb import set_trace as bp

from data.visda17 import VisDA17
from model.resnet import resnet101
from utils.utils import get_params, IterNums, save_checkpoint, AverageMeter, lr_poly, adjust_learning_rate, accuracy
from utils.logger import prepare_logger, prepare_seed

CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
KLDivLoss = nn.KLDivLoss(reduction='batchmean')

parser = argparse.ArgumentParser(description='ASG Training')
parser.add_argument('--data', default='/raid/taskcv-2017-public/classification/data', help='path to dataset')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, dest='batch_size', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--lwf', default=0., type=float, dest='lwf', help='weight of KL loss for LwF (default: 0)')
parser.add_argument('--resume', default='none', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true', help='whether to use learn without forgetting (default: False)')
parser.add_argument('--timestamp', type=str, default='none', help='timestamp for logging naming')
parser.add_argument('--save_dir', type=str, default="./runs", help='root folder to save checkpoints and log.')
parser.add_argument('--train_blocks', type=str, default="conv1.bn1.layer1.layer2.layer3.layer4.fc", help='blocks to train, seperated by dot.')
parser.add_argument('--num-class', default=12, type=int, dest='num_class', help='the number of classes')
parser.add_argument('--rand_seed', default=0, type=int, help='the number of classes')

best_prec1 = 0

def main():
    global args, best_prec1
    PID = os.getpid()
    args = parser.parse_args()
    prepare_seed(args.rand_seed)

    if args.timestamp == 'none':
        args.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    # Log outputs
    if args.evaluate:
        args.save_dir = args.save_dir + "/Visda17-Res101-evaluate" + \
            "%s/%s"%('/'+args.resume if args.resume != 'none' else '', args.timestamp)
    else:
        args.save_dir = args.save_dir + \
            "/Visda17-Res101-%s-train.%s-LR%.2E-epoch%d-batch%d-seed%d"%(
                   "LWF" if args.lwf > 0 else "XE", args.train_blocks, args.lr, args.epochs, args.batch_size, args.rand_seed) + \
            "%s/%s"%('/'+args.resume if args.resume != 'none' else '', args.timestamp)
    logger = prepare_logger(args)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    kwargs = {'num_workers': 20, 'pin_memory': True}
    trainset = VisDA17(txt_file=os.path.join(args.data, "train/image_list.txt"), root_dir=os.path.join(args.data, "train"), transform=data_transforms['train'])
    valset = VisDA17(txt_file=os.path.join(args.data, "validation/image_list.txt"), root_dir=os.path.join(args.data, "validation"), transform=data_transforms['val'], label_one_hot=True)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = resnet101(pretrained=True)
    num_ftrs = model.fc.in_features
    fc_layers = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.num_class),
    )
    model.fc_new = fc_layers

    train_blocks = args.train_blocks.split('.')
    # default turn-off fc, turn-on fc_new
    for param in model.fc.parameters():
        param.requires_grad = False
    ##### Freeze several bottom layers (Optional) #####
    non_train_blocks = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for name in train_blocks:
        try:
            non_train_blocks.remove(name)
        except Exception:
            print("cannot find block name %s\nAvailable blocks are: conv1, bn1, layer1, layer2, layer3, layer4, fc"%name)
    for name in non_train_blocks:
        for param in getattr(model, name).parameters():
            param.requires_grad = False

    # Setup optimizer
    factor = 0.1
    sgd_in = []
    for name in train_blocks:
        if name != 'fc':
            sgd_in.append({'params': get_params(model, [name]), 'lr': factor*args.lr})
        else:
            sgd_in.append({'params': get_params(model, ["fc_new"]), 'lr': args.lr})
    base_lrs = [ group['lr'] for group in sgd_in ]
    optimizer = torch.optim.SGD(sgd_in, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume != 'none':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    model = model.cuda()

    model_old = None
    if args.lwf > 0:
        # create a fixed model copy for Life-long learning
        model_old = resnet101(pretrained=True)
        for param in model_old.parameters():
            param.requires_grad = False
        model_old.eval()
        model_old.cuda()

    if args.evaluate:
        prec1 = validate(val_loader, model)
        print(prec1)
        exit(0)

    # Main training loop
    iter_max = args.epochs * len(train_loader)
    iter_stat = IterNums(iter_max)
    for epoch in range(args.start_epoch, args.epochs):
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save_dir))
        logger.log("Epoch: %d"%(epoch+1))
        # train for one epoch
        train(train_loader, model, optimizer, base_lrs, iter_stat, epoch, logger.writer, model_old=model_old, adjust_lr=True)

        # evaluate on validation set
        prec1 = validate(val_loader, model)
        logger.writer.add_scalar("prec", prec1, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(args.save_dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

    logging.info('Best accuracy: {prec1:.3f}'.format(prec1=best_prec1))


def train(train_loader, model, optimizer, base_lrs, iter_stat, epoch, writer, model_old=None, adjust_lr=True):
    kl_weight = args.lwf
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()

    model.eval()

    # start timer
    end = time.time()

    # train for one epoch
    optimizer.zero_grad()
    epoch_size = len(train_loader)
    train_loader_iter = iter(train_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(epoch_size), file=sys.stdout, bar_format=bar_format, ncols=80)

    for idx_iter in pbar:

        optimizer.zero_grad()
        if adjust_lr:
            lr = lr_poly(base_lrs[-1], iter_stat.iter_curr, iter_stat.iter_max, 0.9)
            writer.add_scalar("lr", lr, idx_iter + epoch * epoch_size)
            adjust_learning_rate(base_lrs, optimizer, iter_stat.iter_curr, iter_stat.iter_max, 0.9)

        input, label = next(train_loader_iter)
        label = label.cuda()
        input = input.cuda()

        # compute output
        output, features_new = model(input, output_features=['layer1', 'layer4'], task='new')

        # compute gradient
        loss = CrossEntropyLoss(output, label.long())

        # LWF KL div
        if model_old is None:
            loss_kl = 0
        else:
            output_new = model.forward_fc(features_new['layer4'], task='old')
            output_old, features_old = model_old(input, output_features=['layer1', 'layer4'], task='old')
            loss_kl = KLDivLoss(F.log_softmax(output_new, dim=1), F.softmax(output_old, dim=1)).sum(-1)

        (loss + kl_weight * loss_kl).backward()

        # measure accuracy and record loss
        losses.update(loss, input.size(0))
        losses_kl.update(loss_kl, input.size(0))

        # compute gradient and do SGD step
        optimizer.step()
        # increment iter number
        iter_stat.update()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar("loss/ce", losses.val, idx_iter + epoch * epoch_size)
        writer.add_scalar("loss/kl", losses_kl.val, idx_iter + epoch * epoch_size)
        writer.add_scalar("loss/total", losses.val + losses_kl.val, idx_iter + epoch * epoch_size)
        description = "[loss: %.3f][loss_kl: %.3f]"%(losses.val, losses_kl.val)
        pbar.set_description("[Step %d/%d]"%(idx_iter + 1, epoch_size) + description)


def validate(val_loader, model):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    val_size = len(val_loader)
    val_loader_iter = iter(val_loader)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(val_size), file=sys.stdout, bar_format=bar_format, ncols=140)
    with torch.no_grad():
        for idx_iter in pbar:
            input, label = next(val_loader_iter)

            input = input.cuda()
            label = label.cuda()

            # compute output
            output = torch.sigmoid(model(input, task='new')[0])
            output = (output + torch.sigmoid(model(torch.flip(input, dims=(3,)), task='new')[0])) / 2

            # accumulate accuracyk
            prec1, gt_num = accuracy(output.data, label, args.num_class, topk=(1,))
            top1.update(prec1[0], gt_num[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            description = "[Acc@1-mean: %.2f][Acc@1-cls: %s]"%(top1.vec2sca_avg, str(top1.avg.numpy().round(1)))
            pbar.set_description("[Step %d/%d]"%(idx_iter + 1, val_size) + description)

    logging.info(' * Prec@1 {top1.vec2sca_avg:.3f}'.format(top1=top1))
    logging.info(' * Prec@1 {top1.avg}'.format(top1=top1))

    return top1.vec2sca_avg


if __name__ == "__main__":
    main()
