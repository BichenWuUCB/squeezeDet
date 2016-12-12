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

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import pascal_voc, kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeDet/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


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
        [mc.CLASS_NAMES[idx] for idx in labels[i]],
        (0, 255, 0))

    # draw prediction
    det_bbox, det_prob, det_class = model.filter_prediction(
        batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] > mc.PLOT_PROB_THRESH]
    det_bbox    = [det_bbox[idx] for idx in keep_idx]
    det_prob    = [det_prob[idx] for idx in keep_idx]
    det_class   = [det_class[idx] for idx in keep_idx]

    _draw_box(
        images[i], det_bbox,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(det_class, det_prob)],
        (0, 0, 255))


def train():
  """Train SqueezeDet model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'

  with tf.Graph().as_default():

    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    if FLAGS.net == 'vgg16':
      mc = kitti_vgg16_config()
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = VGG16ConvDet(mc, FLAGS.gpu)
    elif FLAGS.net == 'resnet50':
      mc = kitti_res50_config()
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = ResNet50ConvDet(mc, FLAGS.gpu)
    elif FLAGS.net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
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
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
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

      if mc.DEBUG_MODE:
        print ('Warning: Discarded {}/({}) labels that are assigned to the same'
               'anchor'.format(num_discarded_labels, num_labels))

      feed_dict = {
          model.image_input: image_per_batch,
          model.keep_prob: mc.KEEP_PROB,
          model.input_mask: np.reshape(
              sparse_to_dense(
                  mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                  [1.0]*len(mask_indices)),
              [mc.BATCH_SIZE, mc.ANCHORS, 1]),
          model.box_delta_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_delta_values),
          model.box_input: sparse_to_dense(
              bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
              box_values),
          model.labels: sparse_to_dense(
              label_indices, 
              [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
              [1.0]*len(label_indices)),
      }

      if step % FLAGS.summary_step == 0:
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

        num_discarded_labels_op = tf.scalar_summary(
            'counter/num_discarded_labels', num_discarded_labels)
        num_labels_op = tf.scalar_summary(
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
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
