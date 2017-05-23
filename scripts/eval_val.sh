#!/bin/bash

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
python ./src/eval.py \
  --dataset=NEXAREAR \
  --data_path=/opt/data/NEXAREAR/ \
  --image_set=val \
  --eval_dir=/opt/data/KITTI/SqueezeDet/eval_val \
  --checkpoint_path=/opt/logs/NEXAREAR/SqueezeDet/train \
  --net=squeezeDet \
  --gpu=0

# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=val \
#   --eval_dir=/tmp/bichen/logs/SqueezeDetPlus/eval_val \
#   --checkpoint_path=/tmp/bichen/logs/SqueezeDetPlus/train \
#   --net=squeezeDet+ \
#   --gpu=0

# =========================================================================== #
# command for vgg16:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=val \
#   --eval_dir=/tmp/bichen/logs/vgg16/eval_val \
#   --checkpoint_path=/tmp/bichen/logs/vgg16/train \
#   --net=squeezeDet+ \
#   --gpu=0

# =========================================================================== #
# command for resnet50:
# =========================================================================== #
# python ./src/eval.py \
#   --dataset=KITTI \
#   --data_path=./data/KITTI \
#   --image_set=val \
#   --eval_dir=/tmp/bichen/logs/resnet50/eval_train \
#   --checkpoint_path=/tmp/bichen/logs/resnet50/train \
#   --net=resnet50 \
#   --gpu=0
