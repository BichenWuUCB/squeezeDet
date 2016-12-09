# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Yolo demo. 

Detect object from a given image, plot the bounding box on the image.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import math
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import model_config
from yolo_model import YoloModel

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('demo_img', 
                           '/home/eecs/bichen/Proj/YOLO/tf-yolo/data/dog.jpg',
                           """Image file to run demo on.""")
tf.app.flags.DEFINE_string('model_checkpoint_path', '/tmp/cifar10_eval',
                           """Checkpoint file.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/yolo_eval',
                           """Directory to write event logs.""")


def demo():
  """Evaluate yolo_model on sample images"""
  mc = model_config()

  img = cv2.imread(FLAGS.demo_img)
  orig_h, orig_w, _ = img.shape
  orig_img = img.copy()

  img = cv2.resize(img, (mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH))
  img = np.asarray(img, dtype=np.float32).reshape(
      (1, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3))
  img -= mc.BGR_MEANS

  with tf.Graph().as_default() as g:
    yolo = YoloModel(mc)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      # saver.restore(sess, FLAGS.model_checkpoint_path)
      # TODO(bichen): this step should load the model checkpoint
      sess.run(tf.initialize_all_variables())

      summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

      prediction = sess.run(
          yolo.preds, feed_dict={yolo.image_input:img, yolo.keep_prob:1})

    results = yolo.interpret_prediction(np.reshape(prediction, [-1]))

    orig_img = orig_img[:,:,(2,1,0)]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12,12))
    ax.imshow(orig_img)

    for i in range(len(results)):
      x = results[i][1]*orig_w
      y = results[i][2]*orig_h
      w = results[i][3]*orig_w
      h = results[i][4]*orig_h

      ax.add_patch(
          plt.Rectangle(
              (x-w/2, y-h/2),
              w, h, fill=False,
              edgecolor='red', linewidth=3.5)
      )
      ax.text(x-w/2, y-h/2-2,
          '{:s} {:.3f}'.format(results[i][0], results[i][-1]),
          bbox=dict(facecolor='blue', alpha=0.5),
          fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(argv=None):
  demo()

if __name__ == '__main__':
    tf.app.run()
