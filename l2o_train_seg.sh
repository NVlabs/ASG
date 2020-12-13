# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

python l2o_train_seg.py \
--epochs 50 \
--batch-size 6 \
--lr 1e-3 \
--num-class 19 \
--gpus 0 \
--gamma 0 \
--early-stop 2 \
--lwf 75. \
--algo reinforce
