#!/bin/bash

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/eval.py \
  --dataset=KITTI \
  --data_path=./data/KITTI \
  --image_set=train \
  --eval_dir=/tmp/bichen/logs/SqueezeDet/eval_train \
  --checkpoint_dir=./data/model_checkpoints/squeezeDet/ \
  --net=squeezeDet \
  --gpu=0

# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=train \
#   --eval_dir=/tmp/bichen/logs/SqueezeDetPlus/eval_train \
#   --checkpoint_dir=./data/model_checkpoints/squeezeDetPlus/ \
#   --net=squeezeDet+ \
#   --gpu=0

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=train \
#   --eval_dir=/tmp/bichen/logs/vgg16/eval_train \
#   --checkpoint_dir=./data/model_checkpoints/vgg16/ \
#   --net=vgg16 \
#   --gpu=0

# =========================================================================== #
# command for resnet50:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=train \
#   --eval_dir=/tmp/bichen/logs/resnet50/eval_train \
#   --checkpoint_dir=./data/model_checkpoints/resnet50/ \
#   --net=resnet50 \
#   --gpu=0
