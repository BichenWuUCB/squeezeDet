# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet+ model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeDetPlus(ModelSkeleton):
  def __init__(self, mc, gpu_id):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=96, size=7, stride=2,
        padding='VALID', freeze=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='VALID')

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=96, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=96, e1x1=64, e3x3=64, freeze=False)
    fire4 = self._fire_layer(
        'fire4', fire3, s1x1=192, e1x1=128, e3x3=128, freeze=False)
    pool4 = self._pooling_layer(
        'pool4', fire4, size=3, stride=2, padding='VALID')

    fire5 = self._fire_layer(
        'fire5', pool4, s1x1=192, e1x1=128, e3x3=128, freeze=False)
    fire6 = self._fire_layer(
        'fire6', fire5, s1x1=288, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=288, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=384, e1x1=256, e3x3=256, freeze=False)
    pool8 = self._pooling_layer(
        'pool8', fire8, size=3, stride=2, padding='VALID')

    fire9 = self._fire_layer(
        'fire9', pool8, s1x1=384, e1x1=256, e3x3=256, freeze=False)

    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=384, e1x1=256, e3x3=256, freeze=False)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=384, e1x1=256, e3x3=256, freeze=False)
    dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', freeze=freeze)

    return tf.concat(3, [ex1x1, ex3x3], name=layer_name+'/concat')
