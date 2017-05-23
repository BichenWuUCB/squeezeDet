# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from config import *
from dataset import kitti, nexarear
from nets import *
from six.moves import xrange
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform

JSON_PREFIX = '.json'

import nx_commons.model_evaulation.iou_engine as iou_engine
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'NEXAREAR',
                           """Currently support KITTI and NEXAREAR datasets.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                            """VOC challenge year. 2007 or 2012"""
                            """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('train_dir', '/opt/data/logs/NEXAREAR/squeezeDet/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_string('test_dir', '/opt/data/logs/NEXAREAR/squeezeDet/test',
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
tf.app.flags.DEFINE_string('pred_json_folder', '/opt/data/NEXAREAR/pred_labels',
                            """Directory where to write pred jsons """)
tf.app.flags.DEFINE_float('iou_threshold', 0.5,
                            """IOU threshold""")


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
  assert FLAGS.dataset == 'KITTI' or FLAGS.dataset == 'NEXAREAR', \
      'Currently only support KITTI and NEXAREAR datasets'

  with tf.Graph().as_default():

    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    if FLAGS.dataset == 'KITTI':
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
    elif FLAGS.dataset == 'NEXAREAR':
        assert FLAGS.net == 'squeezeDet', \
            'Currently only the squeezeDet model is supported for the NEXAREAR dataset'
        if FLAGS.net == 'squeezeDet':
          mc = nexarear_squeezeDet_config()
          mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
          model = SqueezeDet(mc, FLAGS.gpu)
        imdb = nexarear(FLAGS.image_set, FLAGS.data_path, mc)

    if not os.path.isdir(FLAGS.train_dir):
      print(os.makedirs(FLAGS.train_dir))
    model_metric_fname = os.path.join(FLAGS.train_dir, 'model_metrics.txt')
    print('Model metric filename {}'.format(model_metric_fname))
    # save model size, flops, activations by layers
    with open(model_metric_fname, 'w') as f:
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
      print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    test_summary_writer = tf.summary.FileWriter(FLAGS.test_dir, sess.graph)

    precision = tf.placeholder(tf.float32, name='precision')
    precision_op = tf.summary.scalar('precision_summary', precision)

    num_of_detections = tf.placeholder(tf.int32, name='num_of_detections')
    num_of_detections_op = tf.summary.scalar('num_of_detections_summary', num_of_detections)

    localization_error_precentage = tf.placeholder(tf.float32, name='localization_error_precentage')
    localization_error_precentage_op = tf.summary.scalar('localization_error_precentage_summary', localization_error_precentage)

    classification_error_precentage = tf.placeholder(tf.float32, name='classification_error_precentage')
    classification_error_precentage_op = tf.summary.scalar('classification_error_precentage_summary', classification_error_precentage)

    background_error_precentage = tf.placeholder(tf.float32, name='background_error_precentage')
    background_error_precentage_op = tf.summary.scalar('background_error_precentage_summary', background_error_precentage)

    repeated_error_precentage = tf.placeholder(tf.float32, name='repeated_error_precentage')
    repeated_error_precentage_op = tf.summary.scalar('repeated_error_precentage_summary', repeated_error_precentage)

    recall = tf.placeholder(tf.float32, name='recall')
    recall_op = tf.summary.scalar('recall_summary', recall)

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
        test_set = imdb.get_test_set()
        ground_truth_boxes_directory = imdb.get_label_path()
        images_directory = imdb.get_images_path()
        if os.path.isdir(FLAGS.pred_json_folder):
            shutil.rmtree(FLAGS.pred_json_folder)
        os.makedirs(FLAGS.pred_json_folder)
        print('Processing {} test images'.format(len(test_set)))
        for test_image_basename in test_set:
            test_image_fname = os.path.join(images_directory,test_image_basename)
            compute_time, json_dict_res = infer_bounding_boxes(model, sess, test_image_fname, FLAGS.pred_json_folder)

        # extract score
        results = iou_engine.get_bbox_average_iou_evaulation(ground_truth_boxes_directory, FLAGS.pred_json_folder, imdb.get_classes(), FLAGS.iou_threshold , in_images_dir=None, out_images_and_boxes_dir=None)
        print('Model Eval Score {}'.format(results))

        model_eval_summary_feed_dict = {num_of_detections:results['num_of_detections'],precision : results['precision'],localization_error_precentage:results['localization_error_precentage'],
                           classification_error_precentage:results['classification_error_precentage'], background_error_precentage:results['background_error_precentage'],
                           repeated_error_precentage: results['repeated_error_precentage'],recall:results['recall']}
        model_validation_summary = sess.run(
            [num_of_detections_op, precision_op, localization_error_precentage_op, classification_error_precentage_op,
             background_error_precentage_op, repeated_error_precentage_op, recall_op],feed_dict=model_eval_summary_feed_dict)

        for val_stats in model_validation_summary :
          test_summary_writer.add_summary(val_stats, step)

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
      clean_folders([FLAGS.pred_json_folder])

def clean_folders(folders_lst):
    for folder in folders_lst:
        if os.path.isdir(folder):
            shutil.rmtree(folder)

def infer_bounding_boxes(model, sess, image_file, pred_json_folder):
    img_basename = os.path.basename(image_file)
    json_fname = os.path.join(pred_json_folder, img_basename + JSON_PREFIX)
    im = cv2.imread(image_file)
    width = im.shape[1]
    height = im.shape[0]

    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (model.mc.IMAGE_WIDTH, model.mc.IMAGE_HEIGHT))
    input_image = im - model.mc.BGR_MEANS

    DW = float(float(width) / float(model.mc.IMAGE_WIDTH))
    DH = float(float(height) / float(model.mc.IMAGE_HEIGHT))

    t_start = time.time()

    # Detect
    det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict={model.image_input: [input_image], model.keep_prob: 1.0})

    # Filter
    final_boxes, final_probs, final_class = model.filter_prediction(
        det_boxes[0], det_probs[0], det_class[0])

    compute_time = time.time() - t_start

    keep_idx = [idx for idx in range(len(final_probs)) \
                if final_probs[idx] > model.mc.PLOT_PROB_THRESH]
    final_boxes = [final_boxes[idx] for idx in keep_idx]
    final_probs = [final_probs[idx] for idx in keep_idx]
    final_class = [final_class[idx] for idx in keep_idx]

    box_list = final_boxes
    label_list = [model.mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
         for idx, prob in zip(final_class, final_probs)]

    all_out_boxes = []
    for bbox, label in zip(box_list, label_list):
        cx, cy, w, h = bbox
        xmin_sdet = cx - w / 2
        ymin_sdet = cy - h / 2
        xmax_sdet = cx + w / 2
        ymax_sdet = cy + h / 2

        xmin = float(xmin_sdet * DW)
        ymin = float(ymin_sdet * DH)
        xmax = float(xmax_sdet * DW)
        ymax = float(ymax_sdet * DH)

        class_label = label.split(':')[0]  # text before "CLASS: (PROB)"

        out_box = { "type": "RECT",
                    "label": class_label,
                    "position": "UNDEFINED",
                    "bounding_box_with_pose": {  "p0": { 'x' : xmin, 'y' : ymin},
                                                 "p1": { 'x' : xmax, 'y' : ymax},
                                                 "width": float(w*DW),
                                                 "height": float(h*DH),
                                                 "aspect_ratio": float(w*DW)/float(h*DH),
                                                 "pose": "REAR"
                                                 }
                }
        all_out_boxes.append(out_box)
    json_dict_res = {'img_filename': img_basename, 'bounding_box_object_annotation': all_out_boxes}

    json_file = open(json_fname, 'w')
    json.dump(json_dict_res, json_file, indent=4, sort_keys=True)
    json_file.close()
    return compute_time, json_dict_res

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
