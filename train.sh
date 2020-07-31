# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# [x] train resnet101 with proxy guidance on visda17
# [x] evaluation on visda17

python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--lwf 0.1 \
# --resume pretrained/res101_vista17_best.pth.tar \
# --evaluate
