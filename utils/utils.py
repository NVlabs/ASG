# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

import glob
import os
import shutil
import numpy as np
import torch
from pdb import set_trace as bp


def get_params(model, layers=["layer4"]):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    if isinstance(layers, str):
        layers = [layers]
    b = []
    for layer in layers:
        b.append(getattr(model, layer))

    for i in range(len(b)):
        for k, v in b[i].named_parameters():
            if v.requires_grad:
                yield v


def adjust_learning_rate(base_lrs, optimizer, iter_curr, iter_max, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    num_groups = len(optimizer.param_groups)
    for g in range(num_groups):
        optimizer.param_groups[g]['lr'] = lr_poly(base_lrs[g], iter_curr, iter_max, power)


def lr_poly(base_lr, iter, max_iter, power):
    # return min(0.01+0.99*(float(iter)/100)**2.0, 1.0) * base_lr * ((1-float(iter)/max_iter)**power)  # This is with warm up
    return min(0.01+0.99*(float(iter)/100)**2.0, 1.0) * base_lr * ((1-min(float(iter)/max_iter, 0.8))**power)  # This is with warm up & no smaller than last 20% LR
    # return base_lr * ((1-float(iter)/max_iter)**power)


def save_checkpoint(name, state, is_best, filename='checkpoint.pth.tar', keep_last=1):
    """Saves checkpoint to disk"""
    directory = name
    if not os.path.exists(directory):
        os.makedirs(directory)
    models_paths = list(filter(os.path.isfile, glob.glob(directory + "/epoch*.pth.tar")))
    models_paths.sort(key=os.path.getmtime, reverse=False)
    if len(models_paths) == keep_last:
        for i in range(len(models_paths) + 1 - keep_last):
            os.remove(models_paths[i])
    filename = directory + 'epoch_'+str(state['epoch']) + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/'%(name) + 'model_best.pth.tar')


class IterNums(object):
    def __init__(self, iter_max):
        self.iter_max = iter_max
        self.iter_curr = 0

    def reset(self):
        self.iter_curr = 0

    def update(self):
        self.iter_curr += 1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)


class ROC(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.pred = []
        self.label = []

    def update(self, pred, label):
        assert (self.num_class == pred.shape[0]), "num_class mismatch on input predictions!"
        assert (self.num_class == label.shape[0]), "num_class mismatch on input labels!"
        self.pred.append(pred)
        self.label.append(label)

    def roc_curve(self):
        pred = np.hstack(self.pred)
        label = np.hstack(self.label)
        p = label == 1
        n = ~p
        num_p = np.sum(p, axis=1)
        num_n = np.sum(n, axis=1)
        tpr = np.zeros((self.num_class, 101), np.float32)
        fpr = np.zeros((self.num_class, 101), np.float32)
        for idx in range(101):
            thre = 1 - idx/100.0
            pp = pred > thre
            tp = pp & p
            fp = pp & n
            num_tp = np.sum(tp, axis=1)
            num_fp = np.sum(fp, axis=1)
            tpr[:, idx] = num_tp/(num_p + (num_p == 0))
            fpr[:, idx] = num_fp/(num_n + (num_n == 0))
        return tpr, fpr

    def auc(self, tpr, fpr):
        assert(tpr.shape[0] == fpr.shape[0])
        auc = np.zeros(tpr.shape[0], np.float32)
        for idx in range(tpr.shape[0]):
            auc[idx] = metrics.auc(fpr[idx, :], tpr[idx, :])
        return auc


def accuracy(output, label, num_class, topk=(1,)):
    """Computes the precision@k for the specified values of k, currently only k=1 is supported"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    if len(label.size()) == 2:
        # one_hot label
        _, gt = label.topk(maxk, 1, True, True)
    else:
        gt = label
    pred = pred.t()
    pred_class_idx_list = [pred == class_idx for class_idx in range(num_class)]
    gt = gt.t()
    gt_class_number_list = [(gt == class_idx).sum() for class_idx in range(num_class)]
    correct = pred.eq(gt)

    res = []
    gt_num = []
    for k in topk:
        correct_k = correct[:k].float()
        per_class_correct_list = [correct_k[pred_class_idx].sum(0) for pred_class_idx in pred_class_idx_list]
        per_class_correct_array = torch.tensor(per_class_correct_list)
        gt_class_number_tensor = torch.tensor(gt_class_number_list).float()
        gt_class_zeronumber_tensor = gt_class_number_tensor == 0
        gt_class_number_matrix = torch.tensor(gt_class_number_list).float()
        gt_class_acc = per_class_correct_array.mul_(100.0 / gt_class_number_matrix)
        gt_class_acc[gt_class_zeronumber_tensor] = 0
        res.append(gt_class_acc)
        gt_num.append(gt_class_number_matrix)
    return res, gt_num
