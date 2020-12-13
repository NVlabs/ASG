# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# [x] train vgg16 with proxy guidance on GTA5
# [x] evaluation on Cityscapes

python train_seg.py \
--epochs 50 \
--batch-size 6 \
--lr 1e-3 \
--num-class 19 \
--gpus 0 \
--factor 0.1 \
--lwf 75. \
# --evaluate \
# --resume ./pretrained/vgg16_segmentation_best.pth.tar
