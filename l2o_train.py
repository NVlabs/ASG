# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# [x] train resnet101 with both proxy guidance and L2O policy on visda17

import os
import sys
import time
from collections import deque
import logging
from random import choice
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.visda17 import VisDA17
from model.resnet import resnet101
from utils.utils import get_params, IterNums, save_checkpoint, AverageMeter, lr_poly, accuracy
from utils.logger import prepare_logger, prepare_seed
from utils.sgd import SGD

from reinforce.arguments import get_args
from reinforce.models.policy import Policy

from pdb import set_trace as bp

CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
KLDivLoss = nn.KLDivLoss(reduction='batchmean')


def adjust_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_window_sample(train_loader_iter, train_loader, window_size=1):
    samples = []
    while len(samples) < window_size:
        try:
            sample = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            sample = next(train_loader_iter)
        samples.append(sample)
    return samples, train_loader_iter


def train_step(args, _window_size, train_loader_iter, train_loader, model, optimizer, obs_avg, base_lr, pbar, step, total_steps, model_old=None):
    # if obs_avg: average the observation in the window
    losses = []
    losses_kl = []
    fc_mean = []; fc_std = []
    optimizee_step = []
    for idx in range(_window_size):
        optimizer.zero_grad()
        """Train for one sample on the training set"""
        samples, train_loader_iter = get_window_sample(train_loader_iter, train_loader)
        input, label = samples[0]
        label = label.cuda()
        input = input.cuda()
        # compute output
        output, features_new = model(input, output_features=['layer4'], task='new')
        # compute gradient
        loss = CrossEntropyLoss(output, label.long())
        # LWF KL div
        loss_kl = 0
        if model_old is not None:
            output_new = model.forward_fc(features_new['layer4'], task='old')
            output_old, _ = model_old(input, output_features=[], task='old')
            loss_kl = KLDivLoss(F.log_softmax(output_new, dim=1), F.softmax(output_old, dim=1)).sum(-1)
        (loss + args.lwf * loss_kl).backward()
        # compute gradient and do SGD step
        optimizer.step()
        fc_mean.append(model.fc_new[2].weight.mean().detach())
        fc_std.append(model.fc_new[2].weight.std().detach())
        description = "[step: %.5f][loss: %.1f][loss_kl: %.1f][fc_mean: %.3f][fc_std: %.3f]"%(1. * (step + idx) / total_steps, loss, loss_kl, fc_mean[-1]*1000, fc_std[-1]*1000)
        pbar.set_description("[Step %d/%d]"%(step + idx, total_steps) + description)
        losses.append(loss.detach())
        losses_kl.append(loss_kl.detach())
        optimizee_step.append(1. * (step + idx) / total_steps)
    if obs_avg:
        losses = [sum(losses) / len(losses)]
        losses_kl = [sum(losses_kl) / len(losses_kl)]
        fc_mean = [sum(fc_mean) / len(fc_mean)]
        fc_std = [sum(fc_std) / len(fc_std)]
        optimizee_step = [sum(optimizee_step) / len(optimizee_step)]
    losses = [loss for loss in losses]
    losses_kl = [loss_kl for loss_kl in losses_kl]
    optimizee_step = [torch.tensor(step).cuda() for step in optimizee_step]
    observation = torch.stack(losses + losses_kl + optimizee_step + fc_mean + fc_std, dim=0)
    LRs = torch.Tensor([ group['lr'] / base_lr for group in optimizer.param_groups ]).cuda()
    observation = torch.cat([observation, LRs], dim=0).unsqueeze(0) # (batch=1, feature_size=window_size)
    return train_loader_iter, observation, torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(losses_kl, dim=0)), torch.mean(torch.stack(fc_mean, dim=0)), torch.mean(torch.stack(fc_std, dim=0))


def prepare_optimizee(args, sgd_in_names, obs_shape, hidden_size, actor_critic, current_optimizee_step, prev_optimizee_step):
    prev_optimizee_step += current_optimizee_step
    current_optimizee_step = 0

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
    sgd_in = []
    for name in train_blocks:
        if name != 'fc':
            sgd_in.append({'params': get_params(model, [name]), 'lr': args.lr})
        else:
            sgd_in.append({'params': get_params(model, ["fc_new"]), 'lr': args.lr})
    base_lrs = [ group['lr'] for group in sgd_in ]
    optimizer = SGD(sgd_in, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    model = model.cuda()
    model.eval()
    return model, optimizer, current_optimizee_step, prev_optimizee_step


def main():
    args = get_args()
    PID = os.getpid()
    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save_dir))
    prepare_seed(args.rand_seed)

    if args.timestamp == 'none':
        args.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    torch.set_num_threads(1)

    # Log outputs
    args.save_dir = args.save_dir + \
        "/Visda17-L2O.train.Res101-%s-train.%s-LR%.2E-epoch%d-batch%d-seed%d"%(
               "LWF" if args.lwf > 0 else "XE", args.train_blocks, args.lr, args.epochs, args.batch_size, args.rand_seed) + \
        "%s/%s"%('/'+args.resume if args.resume != 'none' else '', args.timestamp)
    logger = prepare_logger(args)

    best_prec1 = 0

    #### preparation ###########################################
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
    train_loader_iter = iter(train_loader)
    current_optimizee_step, prev_optimizee_step = 0, 0

    model_old = None
    if args.lwf > 0:
        # create a fixed model copy for Life-long learning
        model_old = resnet101(pretrained=True)
        for param in model_old.parameters():
            param.requires_grad = False
        model_old.eval()
        model_old.cuda()
    ############################################################

    ### Agent Settings ########################################
    RANDOM = False # False | True | 'init'
    action_space = np.arange(0, 1.1, 0.1)
    obs_avg = True
    _window_size = 1
    window_size = 1 if obs_avg else _window_size
    window_shrink_size = 20 # larger: controller will be updated more frequently
    sgd_in_names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc_new"]
    coord_size = len(sgd_in_names)
    ob_name_lstm = ["loss", "loss_kl", "step", "fc_mean", "fc_std"]
    ob_name_scalar = []
    obs_shape = (len(ob_name_lstm) * window_size + len(ob_name_scalar) + coord_size, )
    _hidden_size = 20
    hidden_size = _hidden_size * len(ob_name_lstm)
    actor_critic = Policy(coord_size, input_size=(len(ob_name_lstm), len(ob_name_scalar)), action_space=len(action_space), hidden_size=_hidden_size, window_size=window_size)
    actor_critic.cuda()
    actor_critic.eval()

    partial = torch.load(args.agent_load_dir, map_location=lambda storage, loc: storage)
    state = actor_critic.state_dict()
    pretrained_dict = {k: v for k, v in partial.items()}
    state.update(pretrained_dict)
    actor_critic.load_state_dict(state)

    ################################################################

    _min_iter = 10
    # reset optmizee
    model, optimizer, current_optimizee_step, prev_optimizee_step = prepare_optimizee(args, sgd_in_names, obs_shape, hidden_size, actor_critic, current_optimizee_step, prev_optimizee_step)
    epoch_size = len(train_loader)
    total_steps = epoch_size*args.epochs
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(int(epoch_size*args.epochs)), file=sys.stdout, bar_format=bar_format, ncols=100)
    _window_size = max(_min_iter, current_optimizee_step + prev_optimizee_step // window_shrink_size)
    train_loader_iter, obs, loss, loss_kl, fc_mean, fc_std = train_step(args, _window_size, train_loader_iter, train_loader, model, optimizer, obs_avg, args.lr, pbar, current_optimizee_step + prev_optimizee_step, total_steps, model_old=model_old)
    logger.writer.add_scalar("loss/ce", loss, current_optimizee_step + prev_optimizee_step)
    logger.writer.add_scalar("loss/kl", loss_kl, current_optimizee_step + prev_optimizee_step)
    logger.writer.add_scalar("loss/total", loss + loss_kl, current_optimizee_step + prev_optimizee_step)
    logger.writer.add_scalar("fc/mean", fc_mean, current_optimizee_step + prev_optimizee_step)
    logger.writer.add_scalar("fc/std", fc_std, current_optimizee_step + prev_optimizee_step)
    current_optimizee_step += _window_size
    pbar.update(_window_size)
    prev_obs = obs.unsqueeze(0)
    prev_hidden = torch.zeros(actor_critic.net.num_recurrent_layers, 1, hidden_size).cuda()
    for epoch in range(args.epochs):
        print("\n===== Epoch %d / %d ====="%(epoch+1, args.epochs))
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, args.save_dir))
        while current_optimizee_step < epoch_size:
            # Sample actions
            with torch.no_grad():
                if not RANDOM:
                    value, action, action_log_prob, recurrent_hidden_states, distribution = actor_critic.act(prev_obs, prev_hidden, deterministic=False)
                    action = action.squeeze()
                    action_log_prob = action_log_prob.squeeze()
                    value = value.squeeze()
                    for idx in range(len(action)):
                        logger.writer.add_scalar("action/%s"%sgd_in_names[idx], action[idx], current_optimizee_step + prev_optimizee_step)
                        logger.writer.add_scalar("entropy/%s"%sgd_in_names[idx], distribution.distributions[idx].entropy(), current_optimizee_step + prev_optimizee_step)
                        optimizer.param_groups[idx]['lr'] = float(action_space[action[idx]]) * args.lr
                        logger.writer.add_scalar("LR/%s"%sgd_in_names[idx], optimizer.param_groups[idx]['lr'], current_optimizee_step + prev_optimizee_step)
                else:
                    if RANDOM is True or RANDOM == 'init':
                        for idx in range(coord_size):
                            optimizer.param_groups[idx]['lr'] = float(choice(action_space)) * args.lr
                    if RANDOM == 'init':
                        RANDOM = 'done'
                    for idx in range(coord_size):
                        logger.writer.add_scalar("LR/%s"%sgd_in_names[idx], optimizer.param_groups[idx]['lr'], current_optimizee_step + prev_optimizee_step)

            # Obser reward and next obs
            _window_size = max(_min_iter, current_optimizee_step + prev_optimizee_step // window_shrink_size)
            _window_size = min(_window_size, epoch_size - current_optimizee_step)
            train_loader_iter, obs, loss, loss_kl, fc_mean, fc_std = train_step(args, _window_size, train_loader_iter, train_loader, model, optimizer, obs_avg, args.lr, pbar, current_optimizee_step + prev_optimizee_step, total_steps, model_old=model_old)
            logger.writer.add_scalar("loss/ce", loss, current_optimizee_step + prev_optimizee_step)
            logger.writer.add_scalar("loss/kl", loss_kl, current_optimizee_step + prev_optimizee_step)
            logger.writer.add_scalar("loss/total", loss + loss_kl, current_optimizee_step + prev_optimizee_step)
            logger.writer.add_scalar("fc/mean", fc_mean, current_optimizee_step + prev_optimizee_step)
            logger.writer.add_scalar("fc/std", fc_std, current_optimizee_step + prev_optimizee_step)
            current_optimizee_step += _window_size
            pbar.update(_window_size)
            prev_obs = obs.unsqueeze(0)
            if not RANDOM: prev_hidden = recurrent_hidden_states
        prev_optimizee_step += current_optimizee_step
        current_optimizee_step = 0

        # evaluate on validation set
        prec1 = validate(val_loader, model, args)
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


def validate(val_loader, model, args):
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
