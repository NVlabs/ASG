# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

python l2o_train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--num-class 12 \
--lwf 0.1 \
--agent_load_dir /raid/ASG/pretrained/policy_res101_vista17.pth
