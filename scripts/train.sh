#!/bin/bash

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
  --data_path=./data/KITTI \
  --image_set=train \
  --train_dir=/tmp/bichen/logs/SqueezeDet/train \
  --net=squeezeDet \
  --summary_step=100 \
  --checkpoint_step=500 \
  --gpu=0

# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== #
# python ./src/train.py \
#   --dataset=KITTI \
#   --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl \
#   --data_path=./data/KITTI \
#   --image_set=train \
#   --train_dir=/tmp/bichen/logs/SqueezeDetPlus/train \
#   --net=squeezeDet+ \
#   --summary_step=100 \
#   --checkpoint_step=500 \
#   --gpu=0

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
# python ./src/train.py \
#   --dataset=KITTI \
#   --pretrained_model_path=./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl \
#   --data_path=./data/KITTI \
#   --image_set=train \
#   --train_dir=/tmp/bichen/logs/vgg16/train \
#   --net=vgg16 \
#   --summary_step=100 \
#   --checkpoint_step=500 \
#   --gpu=0

# =========================================================================== #
# command for resnet50:
# =========================================================================== #
# python ./src/train.py \
#   --dataset=KITTI \
#   --pretrained_model_path=./data/ResNet/ResNet-50-weights.pkl \
#   --data_path=./data/KITTI \
#   --image_set=train \
#   --train_dir=/tmp/bichen/logs/resnet/train \
#   --net=resnet50 \
#   --summary_step=100 \
#   --checkpoint_step=500 \
#   --gpu=0
