# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time
import argparse

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import *
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from nets import *


def _draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
  assert form == 'center' or form == 'diagonal', \
      'bounding box format not accepted: {}.'.format(form)

  for bbox, label in zip(box_list, label_list):

    if form == 'center':
      bbox = bbox_transform(bbox)

    xmin, ymin, xmax, ymax = [int(b) for b in bbox]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    # draw box
    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
    # draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)

def _viz_prediction_result(model, images, bboxes, labels, batch_det_bbox,
                           batch_det_class, batch_det_prob):
  mc = model.mc

  for i in range(len(images)):
    # draw ground truth
    _draw_box(
        images[i], bboxes[i],
        [mc.dataset.CLASS_NAMES[idx] for idx in labels[i]],
        (0, 255, 0))

    # draw prediction
    det_bbox, det_prob, det_class = model.filter_prediction(
        batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] > mc.post_processing.PLOT_PROB_THRESH]
    det_bbox    = [det_bbox[idx] for idx in keep_idx]
    det_prob    = [det_prob[idx] for idx in keep_idx]
    det_class   = [det_class[idx] for idx in keep_idx]

    _draw_box(
        images[i], det_bbox,
        [mc.dataset.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(det_class, det_prob)],
        (0, 0, 255))


def train(mcfg):
  """Train SqueezeDet model"""
  BATCH_SIZE = mcfg.train.BATCH_SIZE
  train_dir = mcfg.train.CHECKPOINT_DIR

  available_models = { 'squeezeDet' : SqueezeDet,
                       'squeezeDet+' : SqueezeDetPlus,
                       'resnet50' : ResNet50ConvDet }

  available_datasets = { 'NEXAREAR' : nexarear }
  assert mcfg.dataset.DATASET in available_datasets.keys(), \
      'Selected dataset not supported: {}'.format(mcfg.dataset.DATASET)

  with tf.Graph().as_default():

    assert mcfg.base_net in available_models.keys() , \
        'Selected neural net architecture not supported: {}'.format(mcfg.base_net)

    #mc = nexarear_squeezeDet_config()
    #mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
    model = available_models[mcfg.base_net](mcfg,mcfg.gpu_id)
    imdb = available_datasets[mcfg.dataset.DATASET](model.ANCHOR_BOX,mcfg)

    N_ANCHORS = len(model.ANCHOR_BOX)
    N_CLASSES = len(mcfg.dataset.CLASS_NAMES)
    KEEP_PROB = mcfg.train.KEEP_PROB
    DEBUG_MODE = mcfg.DEBUG_MODE
    MAX_STEPS = mcfg.train.MAX_STEPS
    CHECKPOINT_STEP = mcfg.train.CHECKPOINT_STEP
    SUMMARY_STEP = mcfg.train.SUMMARY_STEP


    # save model size, flops, activations by layers
    with open(os.path.join(train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
    print ('Model statistics saved to {}.'.format(
      os.path.join(train_dir, 'model_metrics.txt')))

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in xrange(MAX_STEPS):
      start_time = time.time()

      # read batch input
      image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
          bbox_per_batch = imdb.read_batch()

      label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []
      aidx_set = set()
      num_discarded_labels = 0
      num_labels = 0
      for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
          num_labels += 1
          if (i, aidx_per_batch[i][j]) not in aidx_set:
            aidx_set.add((i, aidx_per_batch[i][j]))
            label_indices.append(
                [i, aidx_per_batch[i][j], label_per_batch[i][j]])
            mask_indices.append([i, aidx_per_batch[i][j]])
            bbox_indices.extend(
                [[i, aidx_per_batch[i][j], k] for k in range(4)])
            box_delta_values.extend(box_delta_per_batch[i][j])
            box_values.extend(bbox_per_batch[i][j])
          else:
            num_discarded_labels += 1

      if DEBUG_MODE:
        print ('Warning: Discarded {}/({}) labels that are assigned to the same'
               'anchor'.format(num_discarded_labels, num_labels))

      feed_dict = {
          model.image_input: image_per_batch,
          model.keep_prob: KEEP_PROB,
          model.input_mask: np.reshape(
              sparse_to_dense(
                  mask_indices, [BATCH_SIZE, N_ANCHORS],
                  [1.0]*len(mask_indices)),
              [BATCH_SIZE, N_ANCHORS, 1]),
          model.box_delta_input: sparse_to_dense(
              bbox_indices, [BATCH_SIZE, N_ANCHORS, 4],
              box_delta_values),
          model.box_input: sparse_to_dense(
              bbox_indices, [BATCH_SIZE, N_ANCHORS, 4],
              box_values),
          model.labels: sparse_to_dense(
              label_indices,
              [BATCH_SIZE, N_ANCHORS, N_CLASSES],
              [1.0]*len(label_indices)),
      }

      if step % SUMMARY_STEP == 0:
        op_list = [
            model.train_op, model.loss, summary_op, model.det_boxes,
            model.det_probs, model.det_class, model.conf_loss,
            model.bbox_loss, model.class_loss
        ]
        _, loss_value, summary_str, det_boxes, det_probs, det_class, conf_loss, \
            bbox_loss, class_loss = sess.run(op_list, feed_dict=feed_dict)

        _viz_prediction_result(
            model, image_per_batch, bbox_per_batch, label_per_batch, det_boxes,
            det_class, det_probs)
        image_per_batch = bgr_to_rgb(image_per_batch)
        viz_summary = sess.run(
            model.viz_op, feed_dict={model.image_to_show: image_per_batch})

        num_discarded_labels_op = tf.summary.scalar(
            'counter/num_discarded_labels', num_discarded_labels)
        num_labels_op = tf.summary.scalar(
            'counter/num_labels', num_labels)

        counter_summary_str = sess.run([num_discarded_labels_op, num_labels_op])

        summary_writer.add_summary(summary_str, step)
        summary_writer.add_summary(viz_summary, step)
        for sum_str in counter_summary_str:
          summary_writer.add_summary(sum_str, step)

        print ('conf_loss: {}, bbox_loss: {}, class_loss: {}'.
            format(conf_loss, bbox_loss, class_loss))
      else:
        _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
            [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
             model.class_loss], feed_dict=feed_dict)

      duration = time.time() - start_time

      assert not np.isnan(loss_value), \
          'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
          'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

      if step % 10 == 0:
        num_images_per_step = BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % CHECKPOINT_STEP == 0 or (step + 1) == MAX_STEPS:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    args = parser.parse_args()

    if not os.path.isfile(args.cfg_file):
        #logger.error('Can not find configuration file : {}'.format(args.cfg_file))
        exit(-1)

    if os.path.isfile(args.cfg_file):
        mc = get_model_config(args.cfg_file)

    if tf.gfile.Exists(mc.train.CHECKPOINT_DIR):
        tf.gfile.DeleteRecursively(mc.train.CHECKPOINT_DIR)
    tf.gfile.MakeDirs(mc.train.CHECKPOINT_DIR)

    res = train(mc)

    exit(res)
