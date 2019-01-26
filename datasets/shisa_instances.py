"""Provides data for the MNIST dataset.
The dataset scripts used to create the dataset can be found at:
tensorflow/models/slim/datasets/download_and_convert_mnist.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'VOC2007-%s.tfrecord'

_SPLITS_TO_SIZES = {'train': 6000, 'val': 6000, 'test': 6000}

_NUM_CLASSES = 10

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [224 x 224 x 3] color image.',
    'label': 'A single integer.',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading MNIST.
  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/object/class/label': tf.FixedLenFeature(
          [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
      # 'image/object/bbox/' : tf.FixedLenFeature(
      #     [1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'image/object/bbox/ymin': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'image/object/bbox/xmin': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'image/object/bbox/ymax': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'image/object/bbox/xmax': tf.FixedLenFeature([1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
      'image/source_file': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/source_video': tf.FixedLenFeature((), tf.string, default_value='')
  }
  #tf.VarLenFeature

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(image_key = 'image/encoded', format_key = 'image/format', shape=None, channels=3),
      'label': slim.tfexample_decoder.Tensor('image/object/class/label', shape=[]),
      'object/bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
      'fname': slim.tfexample_decoder.Tensor('image/source_file', shape=[]),
      'videoname': slim.tfexample_decoder.Tensor('image/source_video', shape=[])
  }
  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  num_record = len(list(tf.python_io.tf_record_iterator(file_pattern)))

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      # num_samples=_SPLITS_TO_SIZES[split_name],
      num_samples=num_record,
      num_classes=_NUM_CLASSES,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      labels_to_names=labels_to_names)