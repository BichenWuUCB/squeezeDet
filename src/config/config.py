# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Base Model configurations"""

import os
import json
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

'''
 
  
  KEEP_PROB     # Probability to keep a node in dropout
  
  BATCH_SIZE  # batch size

  PROB_THRESH         # Only keep boxes with probability higher than this threshold
  PLOT_PROB_THRESH    # Only plot boxes with probability higher than this threshold
  NMS_THRESH          # Bounding boxes with IOU larger than this are going to be removed

 
  LOSS_COEF_CONF      # loss coefficient for confidence regression
  LOSS_COEF_CLASS     # loss coefficient for classification regression
  LOSS_COEF_BBOX      # loss coefficient for bounding box regression

  
  DECAY_STEPS          # reduce step size after this many steps
  LR_DECAY_FACTOR      # multiply the learning rate by this factor
  cfg.LEARNING_RATE    # learning rate
  MOMENTUM             # momentum
  WEIGHT_DECAY         # weight decay

  MAX_GRAD_NORM        # gradients with norm larger than this is going to be clipped.
  
  EPSILON = 1e-16         # a small value used to prevent numerical instability
  EXP_THRESH=1.0          # threshold for safe exponential operation
  BATCH_NORM_EPSILON=1e-5 # small value used in batch normalization to prevent dividing by 0. The
                          # default value here is the same with caffe's default value.
  
  DATA_AUGMENTATION    # Whether to do data augmentation
  DRIFT_X              # The range to randomly shift the image widht
  DRIFT_Y              # The range to randomly shift the image height

  MAX_STEPS : Maximum number of batches to run
  SUMMARY_STEP : Number of steps to save summary
  CHECKPOINT_STEP : Number of steps to save checkpoint
  TRAIN_DIR : Directory for saving checkpoints and log results


  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

'''

def get_model_config(cfg_file):
  with open(cfg_file) as infile:
    cfg = json.load(infile)
  mconfig = edict(cfg)

  if 'LOAD_PRETRAINED_MODEL' not in mconfig.initialization:
    mconfig.initialization.LOAD_PRETRAINED_MODEL = False

  # number of categories to classify
  mconfig.dataset.N_CLASSES = len(mconfig.dataset.CLASS_NAMES)
  mconfig.anchor_boxes.ANCHOR_PER_GRID = 9

  return mconfig

