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
import tempfile
import json
import shutil

import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import *
from dataset import *
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from nets import *
import nx_commons.model_evaulation.iou_engine as iou_engine

JSON_PREFIX = '.json'

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
  available_models = { 'squeezeDet' : SqueezeDet,
                       'squeezeDet+' : SqueezeDetPlus,
                       'resnet50' : ResNet50ConvDet }

  available_datasets = { 'NEXAREAR' : nexarear }
  assert mcfg.dataset.DATASET in available_datasets.keys(), \
      'Selected dataset not supported: {}'.format(mcfg.dataset.DATASET)

  with tf.Graph().as_default():

    assert mcfg.base_net in available_models.keys() , \
        'Selected neural net architecture not supported: {}'.format(mcfg.base_net)

    model = available_models[mcfg.base_net](mcfg)
    imdb = available_datasets[mcfg.dataset.DATASET](model.ANCHOR_BOX,mcfg)

    BATCH_SIZE = mcfg.train.BATCH_SIZE
    checkpoint_dir = mcfg.train.CHECKPOINT_DIR
    train_logs = mcfg.train.LOG_TRAIN_DIR

    N_ANCHORS = len(model.ANCHOR_BOX)
    N_CLASSES = len(mcfg.dataset.CLASS_NAMES)
    KEEP_PROB = mcfg.train.KEEP_PROB
    DEBUG_MODE = mcfg.DEBUG_MODE
    MAX_STEPS = mcfg.train.MAX_STEPS
    CHECKPOINT_STEP = mcfg.train.CHECKPOINT_STEP
    SUMMARY_STEP = mcfg.train.SUMMARY_STEP
    NUM_OF_CHECKPOINTS_TO_KEEP = mcfg.train.NUM_OF_CHECKPOINTS_TO_KEEP

    NUM_OF_TEST_ITERATIONS = mcfg.test.NUM_OF_TEST_ITERATIONS
    test_logs = mcfg.test.LOG_TEST_DIR
    IOU_THRESOLD= mcfg.test.IOU_THRESHOLD

    # save model size, flops, activations by layers
    with open(os.path.join(checkpoint_dir, 'model_metrics.txt'), 'w') as f:
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
      os.path.join(checkpoint_dir, 'model_metrics.txt')))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_OF_CHECKPOINTS_TO_KEEP)
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_logs, sess.graph)
    test_summary_writer = tf.summary.FileWriter(test_logs, sess.graph)

    with tf.variable_scope('Test_Model') as scope:
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
        pred_json_folder = tempfile.mkdtemp()
        # test model
        for ii_test in range(NUM_OF_TEST_ITERATIONS):
          test_img_batch, scales_batch, img_fnames_batch = imdb.read_test_image_batch()
          ground_truth_boxes_directory = imdb.get_label_path()
          test_feed_dict = {
              model.image_input: test_img_batch,
              model.keep_prob: KEEP_PROB,
              model.input_mask: np.reshape(
                  sparse_to_dense(
                      mask_indices, [BATCH_SIZE, N_ANCHORS],
                      [1.0] * len(mask_indices)),
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
                  [1.0] * len(label_indices)),
          }
          compute_time = infer_bounding_boxes_on_image_batch(model, sess, test_feed_dict, img_fnames_batch,
                                                             pred_json_folder)
          print('Processing time batch {} {}'.format(ii_test, compute_time))
          # extract score
        results = iou_engine.get_bbox_average_iou_evaulation(ground_truth_boxes_directory, pred_json_folder,
                                                           imdb.classes, IOU_THRESOLD, in_images_dir=None,
                                                           out_images_and_boxes_dir=None)
        print('Model Eval Score {}'.format(results))
        clean_folders([pred_json_folder])

        model_eval_summary_feed_dict = {num_of_detections: results['num_of_detections'],
                                      precision: results['precision'],
                                      localization_error_precentage: results['localization_error_precentage'],
                                      classification_error_precentage: results['classification_error_precentage'],
                                      background_error_precentage: results['background_error_precentage'],
                                      repeated_error_precentage: results['repeated_error_precentage'],
                                      recall: results['recall']}
        model_validation_summary = sess.run(
          [num_of_detections_op, precision_op, localization_error_precentage_op, classification_error_precentage_op,
           background_error_precentage_op, repeated_error_precentage_op, recall_op],
          feed_dict=model_eval_summary_feed_dict)

        for val_stats in model_validation_summary:
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
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def clean_folders(folders_lst):
    for folder in folders_lst:
        if os.path.isdir(folder):
            shutil.rmtree(folder)

def infer_bounding_boxes_on_image_batch(model, sess, test_feed_dict, img_fnames_batch, pred_json_folder):

    t_start = time.time()

    IMAGE_HEIGHT = model.mc.dataset.IMAGE_HEIGHT
    IMAGE_WIDTH = model.mc.dataset.IMAGE_WIDTH

    det_boxes, det_probs, det_class = sess.run(
        [model.det_boxes, model.det_probs, model.det_class],
        feed_dict=test_feed_dict)
    test_img_batch = test_feed_dict[model.image_input]
    for ii_img, img in enumerate(test_img_batch):
        width = img.shape[1]
        height = img.shape[0]


        DW = float(float(width) / float(IMAGE_WIDTH))
        DH = float(float(height) / float(IMAGE_HEIGHT))

        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[ii_img], det_probs[ii_img], det_class[ii_img])

        compute_time = time.time() - t_start

        keep_idx = [idx for idx in range(len(final_probs)) \
                    if final_probs[idx] > model.mc.post_processing.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        box_list = final_boxes
        label_list = [model.mc.dataset.CLASS_NAMES[idx] + ': (%.2f)' % prob \
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
        img_basename = os.path.basename(img_fnames_batch[ii_img])
        json_dict_res = {'img_filename': img_basename, 'bounding_box_object_annotation': all_out_boxes}
        json_fname = os.path.join(pred_json_folder, img_basename + JSON_PREFIX)
        json_file = open(json_fname, 'w')
        json.dump(json_dict_res, json_file, indent=4, sort_keys=True)
        json_file.close()
    return compute_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    args = parser.parse_args()

    if not os.path.isfile(args.cfg_file):
        #logger.error('Can not find configuration file : {}'.format(args.cfg_file))
        exit(-1)

    if os.path.isfile(args.cfg_file):
        mc = get_model_config(args.cfg_file)

    if tf.gfile.Exists(mc.train.LOG_TRAIN_DIR):
        tf.gfile.DeleteRecursively(mc.train.LOG_TRAIN_DIR)
    tf.gfile.MakeDirs(mc.train.LOG_TRAIN_DIR)

    if tf.gfile.Exists(mc.test.LOG_TEST_DIR):
        tf.gfile.DeleteRecursively(mc.test.LOG_TEST_DIR)
    tf.gfile.MakeDirs(mc.test.LOG_TEST_DIR)

    if tf.gfile.Exists(mc.train.CHECKPOINT_DIR):
        tf.gfile.DeleteRecursively(mc.train.CHECKPOINT_DIR)
    tf.gfile.MakeDirs(mc.train.CHECKPOINT_DIR)

    res = train(mc)

    exit(res)
