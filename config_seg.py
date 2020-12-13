# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'ASG'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]


"""Data Dir"""
C.dataset_path = "/home/chenwy/"

C.train_img_root = os.path.join(C.dataset_path, "gta5")
C.train_gt_root = os.path.join(C.dataset_path, "gta5")
C.val_img_root = os.path.join(C.dataset_path, "cityscapes")
C.val_gt_root = os.path.join(C.dataset_path, "cityscapes")
C.test_img_root = os.path.join(C.dataset_path, "cityscapes")
C.test_gt_root = os.path.join(C.dataset_path, "cityscapes")

C.train_source = osp.join(C.train_img_root, "gta5_train.txt")
C.train_target_source = osp.join(C.train_img_root, "cityscapes_train_fine.txt")
C.eval_source = osp.join(C.val_img_root, "cityscapes_val_fine.txt")
C.test_source = osp.join(C.test_img_root, "cityscapes_test.txt")

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])
C.down_sampling_train = [1, 1] # first down_sampling then crop
C.down_sampling_val = [1, 1] # first down_sampling then crop
C.gt_down_sampling = 1
C.num_train_imgs = 12403
C.num_eval_imgs = 500

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""
C.lr = 0.01
C.momentum = 0.9
C.weight_decay = 5e-4
C.nepochs = 30
C.niters_per_epoch = 2000
C.num_workers = 16
C.train_scale_array = [0.75, 1, 1.25]
# C.train_scale_array = [1]

"""Eval Config"""
C.eval_stride_rate = 5 / 6
C.eval_scale_array = [1]
C.eval_flip = True
C.eval_base_size = 1024
C.eval_crop_size = 1024
C.eval_height = 1024
C.eval_width = 2048

# GTA5: 1052x1914
C.image_height = 512
C.image_width = 512
C.is_test = False # if True, prediction files for the test set will be generated
C.is_eval = False # if True, the train.py will only do evaluation for once
