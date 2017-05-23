# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""ResNet50+ConvDet model."""

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


class ResNet50ConvDet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
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

    conv1 = self._conv_bn_layer(
        self.image_input, 'conv1', 'bn_conv1', 'scale_conv1', filters=64,
        size=7, stride=2, freeze=True, conv_with_bias=True)
    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='VALID')

    with tf.variable_scope('conv2_x') as scope:
      with tf.variable_scope('res2a'):
        branch1 = self._conv_bn_layer(
            pool1, 'res2a_branch1', 'bn2a_branch1', 'scale2a_branch1',
            filters=256, size=1, stride=1, freeze=True, relu=False)
        branch2 = self._res_branch(
            pool1, layer_name='2a', in_filters=64, out_filters=256,
            down_sample=False, freeze=True)
        res2a = tf.nn.relu(branch1+branch2, 'relu')
      with tf.variable_scope('res2b'):
        branch2 = self._res_branch(
            res2a, layer_name='2b', in_filters=64, out_filters=256,
            down_sample=False, freeze=True)
        res2b = tf.nn.relu(res2a+branch2, 'relu')
      with tf.variable_scope('res2c'):
        branch2 = self._res_branch(
            res2b, layer_name='2c', in_filters=64, out_filters=256,
            down_sample=False, freeze=True)
        res2c = tf.nn.relu(res2b+branch2, 'relu')

    with tf.variable_scope('conv3_x') as scope:
      with tf.variable_scope('res3a'):
        branch1 = self._conv_bn_layer(
            res2c, 'res3a_branch1', 'bn3a_branch1', 'scale3a_branch1',
            filters=512, size=1, stride=2, freeze=True, relu=False)
        branch2 = self._res_branch(
            res2c, layer_name='3a', in_filters=128, out_filters=512,
            down_sample=True, freeze=True)
        res3a = tf.nn.relu(branch1+branch2, 'relu')
      with tf.variable_scope('res3b'):
        branch2 = self._res_branch(
            res3a, layer_name='3b', in_filters=128, out_filters=512,
            down_sample=False, freeze=True)
        res3b = tf.nn.relu(res3a+branch2, 'relu')
      with tf.variable_scope('res3c'):
        branch2 = self._res_branch(
            res3b, layer_name='3c', in_filters=128, out_filters=512,
            down_sample=False, freeze=True)
        res3c = tf.nn.relu(res3b+branch2, 'relu')
      with tf.variable_scope('res3d'):
        branch2 = self._res_branch(
            res3c, layer_name='3d', in_filters=128, out_filters=512,
            down_sample=False, freeze=True)
        res3d = tf.nn.relu(res3c+branch2, 'relu')

    with tf.variable_scope('conv4_x') as scope:
      with tf.variable_scope('res4a'):
        branch1 = self._conv_bn_layer(
            res3d, 'res4a_branch1', 'bn4a_branch1', 'scale4a_branch1',
            filters=1024, size=1, stride=2, relu=False)
        branch2 = self._res_branch(
            res3d, layer_name='4a', in_filters=256, out_filters=1024,
            down_sample=True)
        res4a = tf.nn.relu(branch1+branch2, 'relu')
      with tf.variable_scope('res4b'):
        branch2 = self._res_branch(
            res4a, layer_name='4b', in_filters=256, out_filters=1024,
            down_sample=False)
        res4b = tf.nn.relu(res4a+branch2, 'relu')
      with tf.variable_scope('res4c'):
        branch2 = self._res_branch(
            res4b, layer_name='4c', in_filters=256, out_filters=1024,
            down_sample=False)
        res4c = tf.nn.relu(res4b+branch2, 'relu')
      with tf.variable_scope('res4d'):
        branch2 = self._res_branch(
            res4c, layer_name='4d', in_filters=256, out_filters=1024,
            down_sample=False)
        res4d = tf.nn.relu(res4c+branch2, 'relu')
      with tf.variable_scope('res4e'):
        branch2 = self._res_branch(
            res4d, layer_name='4e', in_filters=256, out_filters=1024,
            down_sample=False)
        res4e = tf.nn.relu(res4d+branch2, 'relu')
      with tf.variable_scope('res4f'):
        branch2 = self._res_branch(
            res4e, layer_name='4f', in_filters=256, out_filters=1024,
            down_sample=False)
        res4f = tf.nn.relu(res4e+branch2, 'relu')

    dropout4 = tf.nn.dropout(res4f, self.keep_prob, name='drop4')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv5', dropout4, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

  def _res_branch(
      self, inputs, layer_name, in_filters, out_filters, down_sample=False,
      freeze=False):
    """Residual branch constructor.

      Args:
        inputs: input tensor
        layer_name: layer name
        in_filters: number of filters in XX_branch2a and XX_branch2b layers.
        out_filters: number of filters in XX_branch2clayers.
        donw_sample: if true, down sample the input feature map 
        freeze: if true, do not change parameters in this layer
      Returns:
        A residual branch output operation.
    """
    with tf.variable_scope('res'+layer_name+'_branch2'):
      stride = 2 if down_sample else 1
      output = self._conv_bn_layer(
          inputs,
          conv_param_name='res'+layer_name+'_branch2a',
          bn_param_name='bn'+layer_name+'_branch2a',
          scale_param_name='scale'+layer_name+'_branch2a',
          filters=in_filters, size=1, stride=stride, freeze=freeze)
      output = self._conv_bn_layer(
          output,
          conv_param_name='res'+layer_name+'_branch2b',
          bn_param_name='bn'+layer_name+'_branch2b',
          scale_param_name='scale'+layer_name+'_branch2b',
          filters=in_filters, size=3, stride=1, freeze=freeze)
      output = self._conv_bn_layer(
          output,
          conv_param_name='res'+layer_name+'_branch2c',
          bn_param_name='bn'+layer_name+'_branch2c',
          scale_param_name='scale'+layer_name+'_branch2c',
          filters=out_filters, size=1, stride=1, freeze=freeze, relu=False)
      return output
