# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Neural network model base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf


def _add_loss_summaries(total_loss):
  """Add summaries for losses
  Generates loss summaries for visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  """
  losses = tf.get_collection('losses')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    tf.scalar_summary(l.op.name, l)

def _variable_on_device(name, shape, initializer, trainable=True):
  """Helper to create a Variable.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # TODO(bichen): fix the hard-coded data type below
  dtype = tf.float32
  if not callable(initializer):
    var = tf.get_variable(name, initializer=initializer, trainable=trainable)
  else:
    var = tf.get_variable(
        name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_device(name, shape, initializer, trainable)
  if wd is not None and trainable:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

class ModelSkeleton:
  """Base class of NN detection models."""
  def __init__(self, mc):
    self.mc = mc

    # image batch input
    self.image_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_input'
    )
    # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
    # 1.0 in evaluation phase
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # A tensor where an element is 1 if the corresponding box is "responsible"
    # for detection an object and 0 otherwise.
    self.input_mask = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 1], name='box_mask')
    # Tensor used to represent bounding box deltas.
    self.box_delta_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_delta_input')
    # Tensor used to represent bounding box coordinates.
    self.box_input = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, 4], name='box_input')
    # Tensor used to represent labels
    self.labels = tf.placeholder(
        tf.float32, [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES], name='labels')
    # Tensor representing the IOU between predicted bbox and gt bbox
    self.ious = tf.Variable(
        initial_value=np.zeros((mc.BATCH_SIZE, mc.ANCHORS)), trainable=False,
        name='iou', dtype=tf.float32
    )

    # model parameters
    self.model_params = []

    # model size counter
    self.model_size_counter = [] # array of tuple of layer name, parameter size
    # flop counter
    self.flop_counter = [] # array of tuple of layer name, flop number
    # activation counter
    self.activation_counter = [] # array of tuple of layer name, output activations
    self.activation_counter.append(('input', mc.IMAGE_WIDTH*mc.IMAGE_HEIGHT*3))


  def _add_forward_graph(self):
    """NN architecture specification."""
    raise NotImplementedError

  def _add_interpretation_graph(self):
    """Interpret NN output."""
    mc = self.mc

    with tf.variable_scope('interpret_output') as scope:
      preds = self.preds

      # probability
      num_class_probs = mc.ANCHOR_PER_GRID*mc.CLASSES
      self.pred_class_probs = tf.reshape(
          tf.nn.softmax(
              tf.reshape(
                  preds[:, :, :, :num_class_probs],
                  [-1, mc.CLASSES]
              )
          ),
          [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
          name='pred_class_probs'
      )
      
      # confidence
      num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs
      self.pred_conf = tf.sigmoid(
          tf.reshape(
              preds[:, :, :, num_class_probs:num_confidence_scores],
              [mc.BATCH_SIZE, mc.ANCHORS]
          ),
          name='pred_confidence_score'
      )

      # bbox_delta
      self.pred_box_delta = tf.reshape(
          preds[:, :, :, num_confidence_scores:],
          [mc.BATCH_SIZE, mc.ANCHORS, 4],
          name='bbox_delta'
      )

      # number of object. Used to normalize bbox and classification loss
      self.num_objects = tf.reduce_sum(self.input_mask, name='num_objects')

    with tf.variable_scope('bbox') as scope:
      with tf.variable_scope('stretching'):
        delta_x, delta_y, delta_w, delta_h = tf.unpack(
            self.pred_box_delta, axis=2)

        anchor_x = mc.ANCHOR_BOX[:, 0]
        anchor_y = mc.ANCHOR_BOX[:, 1]
        anchor_w = mc.ANCHOR_BOX[:, 2]
        anchor_h = mc.ANCHOR_BOX[:, 3]

        box_center_x = tf.identity(
            anchor_x + delta_x * anchor_w, name='bbox_cx')
        box_center_y = tf.identity(
            anchor_y + delta_y * anchor_h, name='bbox_cy')
        box_width = tf.identity(
            anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH),
            name='bbox_width')
        box_height = tf.identity(
            anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH),
            name='bbox_height')

        self._activation_summary(delta_x, 'delta_x')
        self._activation_summary(delta_y, 'delta_y')
        self._activation_summary(delta_w, 'delta_w')
        self._activation_summary(delta_h, 'delta_h')

        self._activation_summary(box_center_x, 'bbox_cx')
        self._activation_summary(box_center_y, 'bbox_cy')
        self._activation_summary(box_width, 'bbox_width')
        self._activation_summary(box_height, 'bbox_height')

      with tf.variable_scope('trimming'):
        xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
            [box_center_x, box_center_y, box_width, box_height])

        # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
        # pixels. Same for y.
        xmins = tf.minimum(
            tf.maximum(0.0, xmins), mc.IMAGE_WIDTH-1.0, name='bbox_xmin')
        self._activation_summary(xmins, 'box_xmin')

        ymins = tf.minimum(
            tf.maximum(0.0, ymins), mc.IMAGE_HEIGHT-1.0, name='bbox_ymin')
        self._activation_summary(ymins, 'box_ymin')

        xmaxs = tf.maximum(
            tf.minimum(mc.IMAGE_WIDTH-1.0, xmaxs), 0.0, name='bbox_xmax')
        self._activation_summary(xmaxs, 'box_xmax')

        ymaxs = tf.maximum(
            tf.minimum(mc.IMAGE_HEIGHT-1.0, ymaxs), 0.0, name='bbox_ymax')
        self._activation_summary(ymaxs, 'box_ymax')

        self.det_boxes = tf.transpose(
            tf.pack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
            (1, 2, 0), name='bbox'
        )

    with tf.variable_scope('IOU'):
      def _tensor_iou(box1, box2):
        with tf.variable_scope('intersection'):
          xmin = tf.maximum(box1[0], box2[0], name='xmin')
          ymin = tf.maximum(box1[1], box2[1], name='ymin')
          xmax = tf.minimum(box1[2], box2[2], name='xmax')
          ymax = tf.minimum(box1[3], box2[3], name='ymax')

          w = tf.maximum(0.0, xmax-xmin, name='inter_w')
          h = tf.maximum(0.0, ymax-ymin, name='inter_h')
          intersection = tf.mul(w, h, name='intersection')

        with tf.variable_scope('union'):
          w1 = tf.sub(box1[2], box1[0], name='w1')
          h1 = tf.sub(box1[3], box1[1], name='h1')
          w2 = tf.sub(box2[2], box2[0], name='w2')
          h2 = tf.sub(box2[3], box2[1], name='h2')

          union = w1*h1 + w2*h2 - intersection

        return intersection/(union+mc.EPSILON) \
            * tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

      self.ious = self.ious.assign(
          _tensor_iou(
              util.bbox_transform(tf.unpack(self.det_boxes, axis=2)),
              util.bbox_transform(tf.unpack(self.box_input, axis=2))
          )
      )
      self._activation_summary(self.ious, 'conf_score')

    with tf.variable_scope('probability') as scope:
      self._activation_summary(self.pred_class_probs, 'class_probs')

      probs = tf.mul(
          self.pred_class_probs,
          tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          name='final_class_prob'
      )

      self._activation_summary(probs, 'final_class_prob')

      self.det_probs = tf.reduce_max(probs, 2, name='score')
      self.det_class = tf.argmax(probs, 2, name='class_idx')

  def _add_loss_graph(self):
    """Define the loss operation."""
    mc = self.mc

    with tf.variable_scope('class_regression') as scope:
      # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
      # add a small value into log to prevent blowing up
      self.class_loss = tf.truediv(
          tf.reduce_sum(
              (self.labels*(-tf.log(self.pred_class_probs+mc.EPSILON))
               + (1-self.labels)*(-tf.log(1-self.pred_class_probs+mc.EPSILON)))
              * self.input_mask * mc.LOSS_COEF_CLASS),
          self.num_objects,
          name='class_loss'
      )
      tf.add_to_collection('losses', self.class_loss)

    with tf.variable_scope('confidence_score_regression') as scope:
      input_mask = tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
      self.conf_loss = tf.reduce_mean(
          tf.reduce_sum(
              tf.square((self.ious - self.pred_conf)) 
              * (input_mask*mc.LOSS_COEF_CONF_POS/self.num_objects
                 +(1-input_mask)*mc.LOSS_COEF_CONF_NEG/(mc.ANCHORS-self.num_objects)),
              reduction_indices=[1]
          ),
          name='confidence_loss'
      )
      tf.add_to_collection('losses', self.conf_loss)
      tf.scalar_summary('mean iou', tf.reduce_sum(self.ious)/self.num_objects)

    with tf.variable_scope('bounding_box_regression') as scope:
      self.bbox_loss = tf.truediv(
          tf.reduce_sum(
              mc.LOSS_COEF_BBOX * tf.square(
                  self.input_mask*(self.pred_box_delta-self.box_delta_input))),
          self.num_objects,
          name='bbox_loss'
      )
      tf.add_to_collection('losses', self.bbox_loss)

    # add above losses as well as weight decay losses to form the total loss
    self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

  def _add_train_graph(self):
    """Define the training operation."""
    mc = self.mc

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                    self.global_step,
                                    mc.DECAY_STEPS,
                                    mc.LR_DECAY_FACTOR,
                                    staircase=True)

    tf.scalar_summary('learning_rate', lr)

    _add_loss_summaries(self.loss)

    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
    grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())

    with tf.variable_scope('clip_gradient') as scope:
      for i, (grad, var) in enumerate(grads_vars):
        grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

    apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads_vars:
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
      self.train_op = tf.no_op(name='train')

  def _add_viz_graph(self):
    """Define the visualization operation."""
    mc = self.mc
    self.image_to_show = tf.placeholder(
        tf.float32, [None, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
        name='image_to_show'
    )
    self.viz_op = tf.image_summary('sample_detection_results',
        self.image_to_show, collections='image_summary',
        max_images=mc.BATCH_SIZE)

  def _conv_bn_layer(
      self, inputs, conv_param_name, bn_param_name, scale_param_name, filters,
      size, stride, padding='SAME', freeze=False, relu=True,
      conv_with_bias=False, stddev=0.001):
    """ Convolution + BatchNorm + [relu] layer. Batch mean and var are treated
    as constant. Weights have to be initialized from a pre-trained model or
    restored from a checkpoint.

    Args:
      inputs: input tensor
      conv_param_name: name of the convolution parameters
      bn_param_name: name of the batch normalization parameters
      scale_param_name: name of the scale parameters
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      conv_with_bias: whether or not add bias term to the convolution output.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """
    mc = self.mc

    with tf.variable_scope(conv_param_name) as scope:
      channels = inputs.get_shape()[3]

      if mc.LOAD_PRETRAINED_MODEL:
        cw = self.caffemodel_weight
        kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
        if conv_with_bias:
          bias_val = cw[conv_param_name][1]
        mean_val   = cw[bn_param_name][0]
        var_val    = cw[bn_param_name][1]
        gamma_val  = cw[scale_param_name][0]
        beta_val   = cw[scale_param_name][1]
      else:
        kernel_val = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        if conv_with_bias:
          bias_val = tf.constant_initializer(0.0)
        mean_val   = tf.constant_initializer(0.0)
        var_val    = tf.constant_initializer(1.0)
        gamma_val  = tf.constant_initializer(1.0)
        beta_val   = tf.constant_initializer(0.0)

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
      self.model_params += [kernel]
      if conv_with_bias:
        biases = _variable_on_device('biases', [filters], bias_val,
                                     trainable=(not freeze))
        self.model_params += [biases]
      gamma = _variable_on_device('gamma', [filters], gamma_val,
                                  trainable=(not freeze))
      beta  = _variable_on_device('beta', [filters], beta_val,
                                  trainable=(not freeze))
      mean  = _variable_on_device('mean', [filters], mean_val, trainable=False)
      var   = _variable_on_device('var', [filters], var_val, trainable=False)
      self.model_params += [gamma, beta, mean, var]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      if conv_with_bias:
        conv = tf.nn.bias_add(conv, biases, name='bias_add')

      conv = tf.nn.batch_normalization(
          conv, mean=mean, variance=var, offset=beta, scale=gamma,
          variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')

      self.model_size_counter.append(
          (conv_param_name, (1+size*size*int(channels))*filters)
      )
      out_shape = conv.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((conv_param_name, num_flops))

      self.activation_counter.append(
          (conv_param_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      if relu:
        return tf.nn.relu(conv)
      else:
        return conv


  def _conv_layer(
      self, layer_name, inputs, filters, size, stride, padding='SAME',
      freeze=False, xavier=False, relu=True, stddev=0.001):
    """Convolutional layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      filters: number of output filters.
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
      freeze: if true, then do not train the parameters in this layer.
      xavier: whether to use xavier weight initializer or not.
      relu: whether to use relu or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A convolutional layer operation.
    """

    mc = self.mc
    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
        bias_val = cw[layer_name][1]
        # check the shape
        if (kernel_val.shape == 
              (size, size, inputs.get_shape().as_list()[-1], filters)) \
           and (bias_val.shape == (filters, )):
          use_pretrained_param = True
        else:
          print ('Shape of the pretrained parameter of {} does not match, '
              'use randomly initialized parameter'.format(layer_name))
      else:
        print ('Cannot find {} in the pretrained model. Use randomly initialized '
               'parameters'.format(layer_name))

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      channels = inputs.get_shape()[3]

      # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
      # shape [h, w, in, out]
      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val , dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      kernel = _variable_with_weight_decay(
          'kernels', shape=[size, size, int(channels), filters],
          wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

      biases = _variable_on_device('biases', [filters], bias_init, 
                                trainable=(not freeze))
      self.model_params += [kernel, biases]

      conv = tf.nn.conv2d(
          inputs, kernel, [1, stride, stride, 1], padding=padding,
          name='convolution')
      conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
  
      if relu:
        out = tf.nn.relu(conv_bias, 'relu')
      else:
        out = conv_bias

      self.model_size_counter.append(
          (layer_name, (1+size*size*int(channels))*filters)
      )
      out_shape = out.get_shape().as_list()
      num_flops = \
        (1+2*int(channels)*size*size)*filters*out_shape[1]*out_shape[2]
      if relu:
        num_flops += 2*filters*out_shape[1]*out_shape[2]
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append(
          (layer_name, out_shape[1]*out_shape[2]*out_shape[3])
      )

      return out
  
  def _pooling_layer(
      self, layer_name, inputs, size, stride, padding='SAME'):
    """Pooling layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      size: kernel size.
      stride: stride
      padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
    Returns:
      A pooling layer operation.
    """

    with tf.variable_scope(layer_name) as scope:
      out =  tf.nn.max_pool(inputs, 
                            ksize=[1, size, size, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
      activation_size = np.prod(out.get_shape().as_list()[1:])
      self.activation_counter.append((layer_name, activation_size))
      return out

  
  def _fc_layer(
      self, layer_name, inputs, hiddens, flatten=False, relu=True,
      xavier=False, stddev=0.001):
    """Fully connected layer operation constructor.

    Args:
      layer_name: layer name.
      inputs: input tensor
      hiddens: number of (hidden) neurons in this layer.
      flatten: if true, reshape the input 4D tensor of shape 
          (batch, height, weight, channel) into a 2D tensor with shape 
          (batch, -1). This is used when the input to the fully connected layer
          is output of a convolutional layer.
      relu: whether to use relu or not.
      xavier: whether to use xavier weight initializer or not.
      stddev: standard deviation used for random weight initializer.
    Returns:
      A fully connected layer operation.
    """
    mc = self.mc

    use_pretrained_param = False
    if mc.LOAD_PRETRAINED_MODEL:
      cw = self.caffemodel_weight
      if layer_name in cw:
        use_pretrained_param = True
        kernel_val = cw[layer_name][0]
        bias_val = cw[layer_name][1]

    if mc.DEBUG_MODE:
      print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

    with tf.variable_scope(layer_name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
        dim = input_shape[1]*input_shape[2]*input_shape[3]
        inputs = tf.reshape(inputs, [-1, dim])
        if use_pretrained_param:
          try:
            # check the size before layout transform
            assert kernel_val.shape == (hiddens, dim), \
                'kernel shape error at {}'.format(layer_name)
            kernel_val = np.reshape(
                np.transpose(
                    np.reshape(
                        kernel_val, # O x (C*H*W)
                        (hiddens, input_shape[3], input_shape[1], input_shape[2])
                    ), # O x C x H x W
                    (2, 3, 1, 0)
                ), # H x W x C x O
                (dim, -1)
            ) # (H*W*C) x O
            # check the size after layout transform
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            # Do not use pretrained parameter if shape doesn't match
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))
      else:
        dim = input_shape[1]
        if use_pretrained_param:
          try:
            kernel_val = np.transpose(kernel_val, (1,0))
            assert kernel_val.shape == (dim, hiddens), \
                'kernel shape error at {}'.format(layer_name)
          except:
            use_pretrained_param = False
            print ('Shape of the pretrained parameter of {} does not match, '
                   'use randomly initialized parameter'.format(layer_name))

      if use_pretrained_param:
        if mc.DEBUG_MODE:
          print ('Using pretrained model for {}'.format(layer_name))
        kernel_init = tf.constant(kernel_val, dtype=tf.float32)
        bias_init = tf.constant(bias_val, dtype=tf.float32)
      elif xavier:
        kernel_init = tf.contrib.layers.xavier_initializer()
        bias_init = tf.constant_initializer(0.0)
      else:
        kernel_init = tf.truncated_normal_initializer(
            stddev=stddev, dtype=tf.float32)
        bias_init = tf.constant_initializer(0.0)

      weights = _variable_with_weight_decay(
          'weights', shape=[dim, hiddens], wd=mc.WEIGHT_DECAY,
          initializer=kernel_init)
      biases = _variable_on_device('biases', [hiddens], bias_init)
      self.model_params += [weights, biases]
  
      outputs = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
      if relu:
        outputs = tf.nn.relu(outputs, 'relu')

      # count layer stats
      self.model_size_counter.append((layer_name, (dim+1)*hiddens))

      num_flops = 2 * dim * hidden + hidden
      if relu:
        num_flops += 2*hiddens
      self.flop_counter.append((layer_name, num_flops))

      self.activation_counter.append((layer_name, hiddens))

      return outputs

  def filter_prediction(self, boxes, probs, cls_idx):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.

    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """
    mc = self.mc

    if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
    else:
      filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]

    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(mc.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
      keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)
    return final_boxes, final_probs, final_cls_idx

  def _activation_summary(self, x, layer_name):
    """Helper to create summaries for activations.

    Args:
      x: layer output tensor
      layer_name: name of the layer
    Returns:
      nothing
    """
    with tf.variable_scope('activation_summary') as scope:
      tf.histogram_summary(
          'activation_summary/'+layer_name, x)
      tf.scalar_summary(
          'activation_summary/'+layer_name+'/sparsity', tf.nn.zero_fraction(x))
      tf.scalar_summary(
          'activation_summary/'+layer_name+'/average', tf.reduce_mean(x))
      tf.scalar_summary(
          'activation_summary/'+layer_name+'/max', tf.reduce_max(x))
      tf.scalar_summary(
          'activation_summary/'+layer_name+'/min', tf.reduce_min(x))
