# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import os
import sys
import logging
from random import choice
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data.gta5 import GTA5
from data.cityscapes import Cityscapes
from model.vgg import vgg16
from model.fcn8s_vgg import FCN8sAtOnce as FCN_Vgg
from dataloader_seg import get_train_loader
from eval_seg import SegEvaluator
from utils.utils import get_params, IterNums, save_checkpoint, AverageMeter, lr_poly, adjust_learning_rate

from reinforce import algo, utils
from reinforce.arguments import get_args
from reinforce.models.policy import Policy
from reinforce.storage import RolloutStorage

from pdb import set_trace as bp

CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
KLDivLoss = nn.KLDivLoss(reduction='batchmean')


def adjust_lr(optimizer, base_lr, iter_curr, total_iter, power=0.9):
    ''' assume lr in optimizer is raw without warmup '''
    ratio = min(0.01+0.99*(float(iter_curr)/100.)**2.0, 1.0) * ((1-min(float(iter_curr)/total_iter, 0.8))**power)  # This is with warm up & no smaller than last 20% LR
    num_groups = len(optimizer.param_groups)
    for g in range(num_groups):
        optimizer.param_groups[g]['lr'] = base_lr[g] * ratio


def get_window_sample(train_loader_iter, train_loader, window_size=1):
    samples = []
    while len(samples) < window_size:
        try:
            sample = next(train_loader_iter)
        except Exception:
            train_loader_iter = iter(train_loader)
            sample = next(train_loader_iter)
        samples.append(sample)
    return samples, train_loader_iter


def train_step(args, _window_size, train_loader_iter, train_loader, model, optimizer, obs_avg, base_lr, pbar, step, total_steps, model_old=None):
    ##### base lr #####
    action_curr = []
    for g in range(len(optimizer.param_groups)):
        action_curr.append(optimizer.param_groups[g]['lr'])
    ###################
    model.eval()
    losses = []
    losses_kl = []
    fc_mean = []; fc_std = []
    optimizee_step = []
    for idx in range(_window_size):
        ##### warmup #####
        adjust_lr(optimizer, action_curr, step + idx + 1, total_steps)
        ##################
        optimizer.zero_grad()
        """Train for one sample on the training set"""
        samples, train_loader_iter = get_window_sample(train_loader_iter, train_loader)
        label = samples[0]['label'].cuda()
        input = samples[0]['data'].cuda()
        # compute output
        output, features_new = model(input, output_features=['layer4'], task='new_seg')
        # compute gradient
        loss = CrossEntropyLoss(output, label.long())
        # LWF KL div
        loss_kl = 0
        if model_old is not None:
            ##### whole f4 for KL ######################
            output_new = model_old.forward_fc(features_new['layer4'], task='old')
            output_old, features_old = model_old(input, output_features=[], task='old')
            #############################################
            loss_kl = KLDivLoss(F.log_softmax(output_new, dim=1), F.softmax(output_old, dim=1)).sum(-1)
        ##### control KL weight ##########
        if args.lwf * loss_kl > 9:
            _kl_weight = float((9. / loss_kl).detach().cpu().numpy())
        else:
            _kl_weight = args.lwf
        ##################################
        (loss + _kl_weight * loss_kl).backward()
        # compute gradient and do SGD step
        optimizer.step()
        fc_mean.append(model.upscore8.weight.mean().detach())
        fc_std.append(model.upscore8.weight.std().detach())
        description = "[step: %.5f][loss: %.1f][loss_kl: %.1f][fc_mean: %.3f][fc_std: %.3f]"%(1. * (step + idx) / total_steps, loss, loss_kl, fc_mean[-1]*1000, fc_std[-1]*1000)
        pbar.set_description("[Step %d/%d]"%(step + idx, total_steps) + description)
        pbar.update(1)
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
    LRs = torch.Tensor([ action / base_lr for action in action_curr ]).cuda()
    observation = torch.cat([observation, LRs], dim=0).unsqueeze(0) # (batch=1, feature_size=window_size)
    return train_loader_iter, observation, torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(losses_kl, dim=0)), torch.mean(torch.stack(fc_mean, dim=0)), torch.mean(torch.stack(fc_std, dim=0))


def prepare_optimizee(args, sgd_in_names, obs_shape, hidden_size, actor_critic, current_optimizee_step, prev_optimizee_step):
    prev_optimizee_step += current_optimizee_step
    current_optimizee_step = 0

    ##### Vgg16 #####
    vgg = vgg16(pretrained=True)
    model = FCN_Vgg(n_class=args.num_class)
    model.copy_params_from_vgg16(vgg)
    ###################

    # Setup optimizer
    sgd_in = [
        {'params': get_params(model, ["conv1_1", "conv1_2"]), 'lr': args.lr},
        {'params': get_params(model, ["conv2_1", "conv2_2"]), 'lr': args.lr},
        {'params': get_params(model, ["conv3_1", "conv3_2", "conv3_3"]), 'lr': args.lr},
        {'params': get_params(model, ["conv4_1", "conv4_2", "conv4_3"]), 'lr': args.lr},
        {'params': get_params(model, ["conv5_1", "conv5_2", "conv5_3"]), 'lr': args.lr},
        {'params': get_params(model, ["fc6", "fc7"]), 'lr': args.lr},
        {'params': get_params(model, ["score_fr", "score_pool3", "score_pool4", "upscore2", "upscore8", "upscore_pool4"]), 'lr': args.lr},
    ]
    optimizer = torch.optim.SGD(sgd_in, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Specify which GPUs to use
    model = model.cuda()
    model.eval()

    return model, optimizer, current_optimizee_step, prev_optimizee_step


def main():
    args = get_args()
    pid = os.getpid()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    best_mIoU = 0

    #### preparation ###########################################
    from config_seg import config as data_setting
    data_setting.batch_size = args.batch_size
    # if args.src_list is not None:
    #     data_setting.train_source = args.src_list
    # if args.tgt_list is not None:
    #     data_setting.eval_source = args.tgt_list
    train_loader = get_train_loader(data_setting, GTA5, test=False)
    train_loader_iter = iter(train_loader)
    current_optimizee_step, prev_optimizee_step = 0, 0

    model_old = None
    if args.lwf:
        # create a fixed model copy for Life-long learning
        ##### Vgg16 #####
        model_old = vgg16(pretrained=True)
        ###################
        model_old.eval()
        model_old.to(device)
    ############################################################

    ### Agent Settings ########################################
    RANDOM = False # False | True | 'init'
    action_space = np.arange(0, 1.1, 0.1)
    # action_space = np.arange(0, 3); granularity = 0.01
    obs_avg = True
    _window_size = 1
    window_size = 1 if obs_avg else _window_size
    window_shrink_size = 20 # larger: controller will be updated more frequently w.r.t. optimizee_step
    sgd_in_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "FC", "fc_new"]
    coord_size = len(sgd_in_names)
    ob_name_lstm = ["loss", "loss_kl", "step", "fc_mean", "fc_std"]
    ob_name_scalar = []
    obs_shape = (len(ob_name_lstm) * window_size + len(ob_name_scalar) + coord_size, )
    _hidden_size = 20
    hidden_size = _hidden_size * len(ob_name_lstm)
    actor_critic = Policy(coord_size, input_size=(len(ob_name_lstm), len(ob_name_scalar)), action_space=len(action_space), hidden_size=_hidden_size, window_size=window_size)
    actor_critic.to(device)
    actor_critic.eval()

    partial = torch.load("./pretrained/policy_vgg16_segmentation.pth", map_location=lambda storage, loc: storage)
    state = actor_critic.state_dict()
    pretrained_dict = {k: v for k, v in partial.items()}
    state.update(pretrained_dict)
    actor_critic.load_state_dict(state)

    if args.algo == 'reinforce':
        agent = algo.REINFORCE(
            actor_critic,
            args.entropy_coef,
            lr=args.lr_meta,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr_meta,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr_meta,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    ################################################################

    _min_iter = 20
    # reset optmizee
    model, optimizer, current_optimizee_step, prev_optimizee_step = prepare_optimizee(args, sgd_in_names, obs_shape, hidden_size, actor_critic, current_optimizee_step, prev_optimizee_step)

    ##### Logging ###########################
    # Log outputs
    if RANDOM:
        args.name = "Random_GTA5_Min%diter.Step%d.Window%d_batch%d_Epoch%d_LR%.1e.warmpoly_lwf.%d"%\
            (_min_iter, args.num_steps, window_shrink_size, args.batch_size, args.epochs, args.lr, args.lwf)
    else:
        args.name = "metatrain_GTA5_%s.SGD.Gamma%.1f.LRmeta.%.1e.Hidden%d.Loss.avg.exp.Earlystop.%d.Min%diter.Step%d.Window%d_batch%d_Epoch%d_LR%.1e.warmpoly_lwf.%d"%\
            (args.algo, args.gamma, args.lr_meta, _hidden_size, args.early_stop, _min_iter, args.num_steps, window_shrink_size, args.batch_size, args.epochs, args.lr, args.lwf)
        if args.resume:
            args.name += "_resumed"

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Log outputs
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'train.log'
    level = logging.INFO
    format = '%(asctime)s  %(message)s'
    handlers = [logging.FileHandler(filename), logging.StreamHandler()]
    logging.basicConfig(level=level, format=format, handlers=handlers)

    writer = SummaryWriter(directory)
    ###########################################

    threds = 1
    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), args.num_class, np.array([0.485, 0.456, 0.406]),
                    np.array([0.229, 0.224, 0.225]), model, [1, ], False, devices=0, config=data_setting, threds=threds,
                    verbose=False, save_path=None, show_image=False)

    epoch_size = len(train_loader)
    total_steps = epoch_size*args.epochs
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(int(epoch_size*args.epochs)), file=sys.stdout, bar_format=bar_format, ncols=100)
    _window_size = max(_min_iter, current_optimizee_step + prev_optimizee_step // window_shrink_size)
    train_loader_iter, obs, loss, loss_kl, fc_mean, fc_std = train_step(args, _window_size, train_loader_iter, train_loader, model, optimizer, obs_avg, args.lr, pbar, current_optimizee_step + prev_optimizee_step, total_steps, model_old=model_old)
    writer.add_scalar("loss/ce", loss, current_optimizee_step + prev_optimizee_step)
    writer.add_scalar("loss/kl", loss_kl, current_optimizee_step + prev_optimizee_step)
    writer.add_scalar("loss/total", loss + loss_kl, current_optimizee_step + prev_optimizee_step)
    writer.add_scalar("fc/mean", fc_mean, current_optimizee_step + prev_optimizee_step)
    writer.add_scalar("fc/std", fc_std, current_optimizee_step + prev_optimizee_step)
    current_optimizee_step += _window_size
    prev_obs = obs.unsqueeze(0)
    prev_hidden = torch.zeros(actor_critic.net.num_recurrent_layers, 1, hidden_size).cuda()
    for epoch in range(args.epochs):
        print("\n===== Epoch %d / %d ====="%(epoch+1, args.epochs))
        print("============= " + args.name + " ================")
        print("============= PID: " + str(pid) + " ================")
        while current_optimizee_step < epoch_size:
            # Sample actions
            with torch.no_grad():
                if not RANDOM:
                    value, action, action_log_prob, recurrent_hidden_states, distribution = actor_critic.act(prev_obs, prev_hidden, deterministic=False)
                    action = action.squeeze(0)
                    action_log_prob = action_log_prob.squeeze(0)
                    value = value.squeeze(0)
                    for idx in range(len(action)):
                        writer.add_scalar("action/%s"%(sgd_in_names[idx]), action[idx], current_optimizee_step + prev_optimizee_step)
                        writer.add_scalar("entropy/%s"%(sgd_in_names[idx]), distribution.distributions[idx].entropy(), current_optimizee_step + prev_optimizee_step)
                        optimizer.param_groups[idx]['lr'] = float(action_space[action[idx]]) * args.lr
                        writer.add_scalar("LR/%s"%(sgd_in_names[idx]), optimizer.param_groups[idx]['lr'], current_optimizee_step + prev_optimizee_step)
                else:
                    if RANDOM is True or RANDOM == 'init':
                        for idx in range(coord_size):
                            optimizer.param_groups[idx]['lr'] = float(choice(action_space)) * args.lr
                    if RANDOM == 'init':
                        RANDOM = 'done'
                    for idx in range(coord_size):
                        writer.add_scalar("LR/%s"%sgd_in_names[idx], optimizer.param_groups[idx]['lr'], current_optimizee_step + prev_optimizee_step)

            # Obser reward and next obs
            _window_size = max(_min_iter, current_optimizee_step + prev_optimizee_step // window_shrink_size)
            _window_size = min(_window_size, epoch_size - current_optimizee_step)
            train_loader_iter, obs, loss, loss_kl, fc_mean, fc_std = train_step(args, _window_size, train_loader_iter, train_loader, model, optimizer, obs_avg, args.lr, pbar, current_optimizee_step + prev_optimizee_step, total_steps, model_old=model_old)
            writer.add_scalar("loss/ce", loss, current_optimizee_step + prev_optimizee_step)
            writer.add_scalar("loss/kl", loss_kl, current_optimizee_step + prev_optimizee_step)
            writer.add_scalar("loss/total", loss + loss_kl, current_optimizee_step + prev_optimizee_step)
            writer.add_scalar("fc/mean", fc_mean, current_optimizee_step + prev_optimizee_step)
            writer.add_scalar("fc/std", fc_std, current_optimizee_step + prev_optimizee_step)
            current_optimizee_step += _window_size
            prev_obs = obs.unsqueeze(0)
            if not RANDOM: prev_hidden = recurrent_hidden_states
        prev_optimizee_step += current_optimizee_step
        current_optimizee_step = 0

        # evaluate on validation set
        torch.cuda.empty_cache()
        mIoU = validate(evaluator, model)
        writer.add_scalar("mIoU", mIoU, epoch)

        # remember best prec@1 and save checkpoint
        is_best = mIoU > best_mIoU
        best_mIoU = max(mIoU, best_mIoU)
        save_checkpoint(args.name, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mIoU': best_mIoU,
        }, is_best)

        logging.info('Best accuracy: {mIoU:.3f}'.format(mIoU=best_mIoU))


def validate(evaluator, model):
    model.eval()
    # _, mIoU = evaluator.run_online()
    _, mIoU = evaluator.run_online_multiprocess()
    return mIoU


if __name__ == "__main__":
    main()
