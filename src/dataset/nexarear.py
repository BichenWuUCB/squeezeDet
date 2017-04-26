# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Image data base class for kitti"""

import os
import glob
import random
import json
import subprocess
import numpy as np

from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou

class nexarear(imdb):
  def __init__(self, ALL_ANCHOR_BOXES, mc):
    imdb.__init__(self, 'nexarear', ALL_ANCHOR_BOXES, mc)
    self._image_path = mc.dataset.images_dir
    self._label_path = mc.dataset.labels_dir
    self._classes = mc.dataset.CLASS_NAMES
    self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))

    # self._image_idx : a list of file names for train images in the directory
    # self._test_image_idx : a list of file names for test images in the directory
    self._image_idx, self._test_image_idx = self._load_image_set_idx(mc)
    print ('Number of train images: {}'.format(len(self._image_idx)))
    print ('Number of test images: {}'.format(len(self._test_image_idx)))

    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height
    self._rois = self._load_nexarear_annotation()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0

    seed = 414
    self._shuffle_image_idx(seed=seed)

    self._eval_tool = './src/dataset/kitti-eval/cpp/evaluate_object'

  def _load_image_set_idx(self,mc):
    assert os.path.isdir(self._image_path), \
      'Directory for images does not exist: {}'.format(self._image_path)

    IMG_TYPES = ('/*.jpg', '/*.png', '/*.JPG')
    image_fnames = []
    for type in IMG_TYPES:
      image_fnames.extend(glob.glob(self._image_path + type))

    image_base_names = [os.path.basename(im) for im in image_fnames]
    n_in_test_set = int(mc.dataset.PERCENTAGE_OF_TEST_SET * float(len(image_base_names) ))

    random.seed(mc.dataset.RANDOM_SEED_TEST_TRAIN_SPLIT)
    random.shuffle(image_base_names)
    return  image_base_names[n_in_test_set:],image_base_names[:n_in_test_set]


  def _image_path_at(self, img_fname):
    image_path = os.path.join(self._image_path, img_fname)
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path

  def _load_nexarear_annotation(self):
    assert os.path.isdir(self._label_path), \
      'Directory for labels does not exist: {}'.format(self._label_path)

    idx2annotation = {}
    for img_fname in self._image_idx:
      boxes_fname = os.path.join(self._label_path, img_fname +'.json')
      if os.path.isfile(boxes_fname):
        with open(boxes_fname) as infile:
          boxes_in_image = json.load(infile)
      else:
        print ('Label file not found: {}'.format(boxes_fname))
        boxes_in_image = {'bounding_box_object_annotation': []}

      bboxes = []
      for box in boxes_in_image['bounding_box_object_annotation']:
        xmin = box['bounding_box_with_pose']['p0']['x']
        xmax = box['bounding_box_with_pose']['p1']['x']
        ymin = box['bounding_box_with_pose']['p0']['y']
        ymax = box['bounding_box_with_pose']['p1']['y']
        label = box['label'].lower()
        try:
          cls = self._class_to_idx[label.strip()]
        except:
          continue

        assert xmin >= 0.0 and xmin <= xmax, \
            'Invalid bounding box x-coord xmin {} or xmax {} at {}' \
                .format(xmin, xmax, img_fname)
        assert ymin >= 0.0 and ymin <= ymax, \
            'Invalid bounding box y-coord ymin {} or ymax {} at {}' \
                .format(ymin, ymax, img_fname)
        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        bboxes.append([x, y, w, h, cls])

      idx2annotation[img_fname] = bboxes

    return idx2annotation

  def evaluate_detections(self, eval_dir, global_step, all_boxes):
    """Evaluate detection results.
    Args:
      eval_dir: directory to write evaluation logs
      global_step: step of the checkpoint
      all_boxes: all_boxes[cls][image] = N x 5 arrays of 
        [xmin, ymin, xmax, ymax, score]
    Returns:
      aps: array of average precisions.
      names: class names corresponding to each ap
    """
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
    if not os.path.isdir(det_file_dir):
      os.makedirs(det_file_dir)

    for im_idx, index in enumerate(self._image_idx):
      filename = os.path.join(det_file_dir, index+'.txt')
      with open(filename, 'wt') as f:
        for cls_idx, cls in enumerate(self._classes):
          dets = all_boxes[cls_idx][im_idx]
          for k in xrange(len(dets)):
            f.write(
                '{:s} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 '
                '0.0 0.0 {:.3f}\n'.format(
                    cls.lower(), dets[k][0], dets[k][1], dets[k][2], dets[k][3],
                    dets[k][4])
            )

    cmd = self._eval_tool + ' ' \
          + os.path.join(self._data_root_path, 'training') + ' ' \
          + os.path.join(self._data_root_path, 'ImageSets',
                         self._image_set+'.txt') + ' ' \
          + os.path.dirname(det_file_dir) + ' ' + str(len(self._image_idx))

    print('Running: {}'.format(cmd))
    status = subprocess.call(cmd, shell=True)

    aps = []
    names = []
    for cls in self._classes:
      det_file_name = os.path.join(
          os.path.dirname(det_file_dir), 'stats_{:s}_ap.txt'.format(cls))
      if os.path.exists(det_file_name):
        with open(det_file_name, 'r') as f:
          lines = f.readlines()
        assert len(lines) == 3, \
            'Line number of {} should be 3'.format(det_file_name)

        aps.append(float(lines[0].split('=')[1].strip()))
        aps.append(float(lines[1].split('=')[1].strip()))
        aps.append(float(lines[2].split('=')[1].strip()))
      else:
        aps.extend([0.0, 0.0, 0.0])

      names.append(cls+'_easy')
      names.append(cls+'_medium')
      names.append(cls+'_hard')

    return aps, names

  def do_detection_analysis_in_eval(self, eval_dir, global_step):
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
    det_error_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step),
        'error_analysis')
    if not os.path.exists(det_error_dir):
      os.makedirs(det_error_dir)
    det_error_file = os.path.join(det_error_dir, 'det_error_file.txt')

    stats = self.analyze_detections(det_file_dir, det_error_file)
    ims = self.visualize_detections(
        image_dir=self._image_path,
        image_format='.png',
        det_error_file=det_error_file,
        output_image_dir=det_error_dir,
        num_det_per_type=10
    )

    return stats, ims

  def analyze_detections(self, detection_file_dir, det_error_file):
    def _save_detection(f, idx, error_type, det, score):
      f.write(
          '{:s} {:s} {:.1f} {:.1f} {:.1f} {:.1f} {:s} {:.3f}\n'.format(
              idx, error_type,
              det[0]-det[2]/2., det[1]-det[3]/2.,
              det[0]+det[2]/2., det[1]+det[3]/2.,
              self._classes[int(det[4])],
              score
          )
      )

    # load detections
    self._det_rois = {}
    for idx in self._image_idx:
      det_file_name = os.path.join(detection_file_dir, idx+'.txt')
      with open(det_file_name) as f:
        lines = f.readlines()
      f.close()
      bboxes = []
      for line in lines:
        obj = line.strip().split(' ')
        cls = self._class_to_idx[obj[0].lower().strip()]
        xmin = float(obj[4])
        ymin = float(obj[5])
        xmax = float(obj[6])
        ymax = float(obj[7])
        score = float(obj[-1])

        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        bboxes.append([x, y, w, h, cls, score])
      bboxes.sort(key=lambda x: x[-1], reverse=True)
      self._det_rois[idx] = bboxes

    # do error analysis
    num_objs = 0.
    num_dets = 0.
    num_correct = 0.
    num_loc_error = 0.
    num_cls_error = 0.
    num_bg_error = 0.
    num_repeated_error = 0.
    num_detected_obj = 0.

    with open(det_error_file, 'w') as f:
      for idx in self._image_idx:
        gt_bboxes = np.array(self._rois[idx])
        num_objs += len(gt_bboxes)
        detected = [False]*len(gt_bboxes)

        det_bboxes = self._det_rois[idx]
        for i, det in enumerate(det_bboxes):
          if i < len(gt_bboxes):
            num_dets += 1
          ious = batch_iou(gt_bboxes[:, :4], det[:4])
          max_iou = np.max(ious)
          gt_idx = np.argmax(ious)
          # if not detected[gt_idx]:
          #   if max_iou > 0.1:
          #     if gt_bboxes[gt_idx, 4] == det[4]:
          #       if max_iou >= 0.5:
          #         if i < len(gt_bboxes):
          #           num_correct += 1
          #         detected[gt_idx] = True
          #       else:
          #         if i < len(gt_bboxes):
          #           num_loc_error += 1
          #           _save_detection(f, idx, 'loc', det, det[5])
          #     else:
          #       if i < len(gt_bboxes):
          #         num_cls_error += 1
          #         _save_detection(f, idx, 'cls', det, det[5])
          #   else:
          #     if i < len(gt_bboxes):
          #       num_bg_error += 1
          #       _save_detection(f, idx, 'bg', det, det[5])
          # else:
          #   if i < len(gt_bboxes):
          #     num_repeated_error += 1

          if max_iou > 0.1:
            if gt_bboxes[gt_idx, 4] == det[4]:
              if max_iou >= 0.5:
                if i < len(gt_bboxes):
                  if not detected[gt_idx]:
                    num_correct += 1
                    detected[gt_idx] = True
                  else:
                    num_repeated_error += 1
              else:
                if i < len(gt_bboxes):
                  num_loc_error += 1
                  _save_detection(f, idx, 'loc', det, det[5])
            else:
              if i < len(gt_bboxes):
                num_cls_error += 1
                _save_detection(f, idx, 'cls', det, det[5])
          else:
            if i < len(gt_bboxes):
              num_bg_error += 1
              _save_detection(f, idx, 'bg', det, det[5])

        for i, gt in enumerate(gt_bboxes):
          if not detected[i]:
            _save_detection(f, idx, 'missed', gt, -1.0)
        num_detected_obj += sum(detected)
    f.close()

    print ('Detection Analysis:')
    print ('    Number of detections: {}'.format(num_dets))
    print ('    Number of objects: {}'.format(num_objs))
    print ('    Percentage of correct detections: {}'.format(
      num_correct/num_dets))
    print ('    Percentage of localization error: {}'.format(
      num_loc_error/num_dets))
    print ('    Percentage of classification error: {}'.format(
      num_cls_error/num_dets))
    print ('    Percentage of background error: {}'.format(
      num_bg_error/num_dets))
    print ('    Percentage of repeated detections: {}'.format(
      num_repeated_error/num_dets))
    print ('    Recall: {}'.format(
      num_detected_obj/num_objs))

    out = {}
    out['num of detections'] = num_dets
    out['num of objects'] = num_objs
    out['% correct detections'] = num_correct/num_dets
    out['% localization error'] = num_loc_error/num_dets
    out['% classification error'] = num_cls_error/num_dets
    out['% background error'] = num_bg_error/num_dets
    out['% repeated error'] = num_repeated_error/num_dets
    out['% recall'] = num_detected_obj/num_objs

    return out