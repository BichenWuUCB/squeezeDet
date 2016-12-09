# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Image data base class for pascal voc"""

import cv2
import os 
import numpy as np
import xml.etree.ElementTree as ET

from utils.util import bbox_transform_inv
from dataset.imdb import imdb
from dataset.voc_eval import voc_eval

class pascal_voc(imdb):
  def __init__(self, image_set, year, data_path, mc):
    imdb.__init__(self, 'voc_'+year+'_'+image_set, mc)
    self._year = year
    self._image_set = image_set
    self._data_root_path = data_path
    self._data_path = os.path.join(self._data_root_path, 'VOC' + self._year)
    self._classes = self.mc.CLASS_NAMES
    self._class_to_idx = dict(zip(self.classes, xrange(self.num_classes)))

    # a list of string indices of images in the directory
    self._image_idx = self._load_image_set_idx() 
    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height
    self._rois = self._load_pascal_annotation()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

  def _load_image_set_idx(self):
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set+'.txt')
    assert os.path.exists(image_set_file), \
        'File does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
      image_idx = [x.strip() for x in f.readlines()]
    return image_idx

  def _image_path_at(self, idx):
    image_path = os.path.join(self._data_path, 'JPEGImages', idx+'.jpg')
    assert os.path.exists(image_path), \
        'Image does not exist: {}'.format(image_path)
    return image_path

  def _load_pascal_annotation(self):
    idx2annotation = {}
    for index in self._image_idx:
      filename = os.path.join(self._data_path, 'Annotations', index+'.xml')
      tree = ET.parse(filename)
      objs = tree.findall('object')
      objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
      bboxes = []
      for obj in objs:
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        xmin = float(bbox.find('xmin').text) - 1
        xmax = float(bbox.find('xmax').text) - 1
        ymin = float(bbox.find('ymin').text) - 1
        ymax = float(bbox.find('ymax').text) - 1
        assert xmin >= 0.0 and xmin <= xmax, \
            'Invalid bounding box x-coord xmin {} or xmax {} at {}.xml' \
                .format(xmin, xmax, index)
        assert ymin >= 0.0 and ymin <= ymax, \
            'Invalid bounding box y-coord ymin {} or ymax {} at {}.xml' \
                .format(ymin, ymax, index)
        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        cls = self._class_to_idx[obj.find('name').text.lower().strip()]
        bboxes.append([x, y, w, h, cls])

      idx2annotation[index] = bboxes

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
        eval_dir, 'detection_files_{:s}'.format(global_step))
    if not os.path.isdir(det_file_dir):
      os.mkdir(det_file_dir)
    det_file_path_template = os.path.join(det_file_dir, '{:s}.txt')

    for cls_idx, cls in enumerate(self._classes):
      det_file_name = det_file_path_template.format(cls)
      with open(det_file_name, 'wt') as f:
        for im_idx, index in enumerate(self._image_idx):
          dets = all_boxes[cls_idx][im_idx]
          # VOC expects 1-based indices
          for k in xrange(len(dets)):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                format(index, dets[k][-1], 
                       dets[k][0]+1, dets[k][1]+1,
                       dets[k][2]+1, dets[k][3]+1)
            )

    # Evaluate detection results
    annopath = os.path.join(
        self._data_root_path,
        'VOC'+self._year,
        'Annotations',
        '{:s}.xml'
    )
    imagesetfile = os.path.join(
        self._data_root_path,
        'VOC'+self._year,
        'ImageSets',
        'Main',
        self._image_set+'.txt'
    )
    cachedir = os.path.join(self._data_root_path, 'annotations_cache')
    aps = []
    use_07_metric = True if int(self._year) < 2010 else False
    for i, cls in enumerate(self._classes):
      filename = det_file_path_template.format(cls)
      _,  _, ap = voc_eval(
          filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
          use_07_metric=use_07_metric)
      aps += [ap]
      print ('{:s}: AP = {:.4f}'.format(cls, ap))

    print ('Mean AP = {:.4f}'.format(np.mean(aps)))
    return aps, self._classes
