# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

import argparse
import os
import sys
import logging
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from data.gta5 import GTA5
from data.cityscapes import Cityscapes
from model.vgg import vgg16
from model.fcn8s_vgg import FCN8sAtOnce as FCN_Vgg
from dataloader_seg import get_train_loader
from eval_seg import SegEvaluator
from utils.utils import get_params, IterNums, save_checkpoint, AverageMeter, lr_poly, adjust_learning_rate
from pdb import set_trace as bp
torch.backends.cudnn.enabled = True

CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
KLDivLoss = nn.KLDivLoss(reduction='batchmean')
best_mIoU = 0

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=6, type=int, dest='batch_size', help='mini-batch size (default: 6)')
parser.add_argument('--iter-size', default=1, type=int, dest='iter_size', help='iteration size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--lwf', default=0., type=float, dest='lwf', help='weight of KL loss for LwF (default: 0)')
parser.add_argument('--factor', default=0.1, type=float, dest='factor', help='scale factor of backbone learning rate (default: 0.1)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Vgg16_GTA5', type=str, help='name of experiment')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--num-class', default=19, type=int, dest='num_class', help='the number of classes')
parser.add_argument('--gpus', default=0, type=int, help='use gpu with cuda number')
parser.add_argument('--evaluate', action='store_true', help='whether to use learn without forgetting (default: False)')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)


def main():
    global args, best_mIoU
    args = parser.parse_args()
    pid = os.getpid()

    # Log outputs
    args.name = "GTA5_Vgg16_batch%d_512x512_Poly_LR%.1e_1to%.1f_all_lwf.%d_epoch%d"%(args.batch_size, args.lr, args.factor, args.lwf, args.epochs)
    if args.resume:
        args.name += "_resumed"
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'train.log'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    rootLogger = logging.getLogger()
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)

    writer = SummaryWriter(directory)

    from config_seg import config as data_setting
    data_setting.batch_size = args.batch_size
    train_loader = get_train_loader(data_setting, GTA5, test=False)

    ##### Vgg16 #####
    vgg = vgg16(pretrained=True)
    model = FCN_Vgg(n_class=args.num_class)
    model.copy_params_from_vgg16(vgg)
    ###################
    threds = 1
    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), args.num_class, np.array([0.485, 0.456, 0.406]),
                    np.array([0.229, 0.224, 0.225]), model, [1, ], False, devices=args.gpus, config=data_setting, threds=threds,
                    verbose=False, save_path=None, show_image=False)

    # Setup optimizer
    ##### Vgg16 #####
    sgd_in = [
        {'params': get_params(model, ["conv1_1", "conv1_2"]), 'lr': args.factor*args.lr},
        {'params': get_params(model, ["conv2_1", "conv2_2"]), 'lr': args.factor*args.lr},
        {'params': get_params(model, ["conv3_1", "conv3_2", "conv3_3"]), 'lr': args.factor*args.lr},
        {'params': get_params(model, ["conv4_1", "conv4_2", "conv4_3"]), 'lr': args.factor*args.lr},
        {'params': get_params(model, ["conv5_1", "conv5_2", "conv5_3"]), 'lr': args.factor*args.lr},
        {'params': get_params(model, ["fc6", "fc7"]), 'lr': args.factor*args.lr},
        {'params': get_params(model, ["score_fr", "score_pool3", "score_pool4", "upscore2", "upscore8", "upscore_pool4"]), 'lr': args.lr},
    ]
    base_lrs = [ group['lr'] for group in sgd_in ]
    optimizer = torch.optim.SGD(sgd_in, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=ImageClassdata> no checkpoint found at '{}'".format(args.resume))

    model = model.cuda()
    model_old = None
    if args.lwf > 0:
        # create a fixed model copy for Life-long learning
        model_old = vgg16(pretrained=True)
        ###################
        for param in model_old.parameters():
            param.requires_grad = False
        model_old.eval()
        model_old.cuda()

    if args.evaluate:
        mIoU = validate(evaluator, model)
        print(mIoU)

    # Main training loop
    iter_max = args.epochs * math.ceil(len(train_loader)/args.iter_size)
    iter_stat = IterNums(iter_max)
    for epoch in range(args.start_epoch, args.epochs):
        logging.info("============= " + args.name + " ================")
        logging.info("============= PID: " + str(pid) + " ================")
        logging.info("Epoch: %d"%(epoch+1))
        # train for one epoch
        train(args, train_loader, model, optimizer, base_lrs, iter_stat, epoch, writer, model_old=model_old, adjust_lr=epoch<args.epochs)
        # evaluate on validation set
        torch.cuda.empty_cache()
        mIoU = validate(evaluator, model)
        writer.add_scalar("mIoU", mIoU, epoch)
        # remember best mIoU and save checkpoint
        is_best = mIoU > best_mIoU
        best_mIoU = max(mIoU, best_mIoU)
        save_checkpoint(directory, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mIoU': best_mIoU,
        }, is_best)

    logging.info('Best accuracy: {mIoU:.3f}'.format(mIoU=best_mIoU))


def train(args, train_loader, model, optimizer, base_lrs, iter_stat, epoch, writer, model_old=None, adjust_lr=True):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    losses_kl = AverageMeter()

    model.eval()

    # train for one epoch
    optimizer.zero_grad()
    epoch_size = len(train_loader)
    train_loader_iter = iter(train_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(epoch_size), file=sys.stdout, bar_format=bar_format, ncols=80)

    for idx_iter in pbar:
        loss_print = 0
        loss_kl_print = 0
        avg_size = 0

        optimizer.zero_grad()
        if adjust_lr:
            lr = lr_poly(base_lrs[-1], iter_stat.iter_curr, iter_stat.iter_max, 0.9)
            writer.add_scalar("lr", lr, idx_iter + epoch * epoch_size)
            adjust_learning_rate(base_lrs, optimizer, iter_stat.iter_curr, iter_stat.iter_max, 0.9)

        sample = next(train_loader_iter)
        label = sample['label'].cuda()
        input = sample['data'].cuda()

        # compute output
        output, features_new = model(input, output_features=['layer4'], task='new_seg')

        # compute gradient
        loss = CrossEntropyLoss(output, label.long())
        loss_print += loss

        # LWF KL div
        if model_old is None:
            loss_kl = 0
        else:
            output_new = model_old.forward_fc(features_new['layer4'], task='old')
            output_old, features_old = model_old(input, output_features=[], task='old')
            loss_kl = KLDivLoss(F.log_softmax(output_new, dim=1), F.softmax(output_old, dim=1)).sum(-1)
        loss_kl_print += loss_kl

        (loss + args.lwf * loss_kl).backward()

        # update size
        avg_size += input.size(0)

        # measure accuracy and record loss
        losses.update(loss_print, avg_size)
        losses_kl.update(loss_kl_print, avg_size)

        # compute gradient and do SGD step
        optimizer.step()
        # increment iter number
        iter_stat.update()

        writer.add_scalar("loss/ce", losses.val, idx_iter + epoch * epoch_size)
        writer.add_scalar("loss/kl", losses_kl.val, idx_iter + epoch * epoch_size)
        writer.add_scalar("loss/total", losses.val + losses_kl.val, idx_iter + epoch * epoch_size)
        description = "[loss: %.3f][loss_kl: %.3f]"%(losses.val, losses_kl.val)
        pbar.set_description("[Step %d/%d]"%(idx_iter + 1, epoch_size) + description)


def validate(evaluator, model):
    with torch.no_grad():
        model.eval()
        # _, mIoU = evaluator.run_online()
        _, mIoU = evaluator.run_online_multiprocess()
    return mIoU


if __name__ == '__main__':
    main()
