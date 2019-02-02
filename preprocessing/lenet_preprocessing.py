# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

import math
import random
seed1 = random.randint(0, 1000000)
seed2 = random.randint(0, 1000000)
seed3 = random.randint(0, 1000000)
seed4 = random.randint(0, 1000000)
seed5 = random.randint(0, 1000000)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def tf_image_translate(images, tx, ty, interpolation='NEAREST'):
  # got these parameters from solving the equations for pixel translations
  # on https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
  transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
  return tf.contrib.image.transform(images, transforms, interpolation)

def preprocess_image(image, bbox, output_height, output_width, is_training):
  """Preprocesses the given image.
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
  Returns:
    A preprocessed image.
  """
  # tf.summary.image('image_orig', image[tf.newaxis, :], 1)

  cropped = tf.squeeze(tf.image.crop_and_resize(image[tf.newaxis, :], bbox, box_ind=[0], crop_size=[output_height, output_width]), [0])
  # # アス比を保ったまま，縦横幅の小さい方に合わせてcropする
  min_size = tf.reduce_min(tf.shape(image)[:-1])
  image = tf.image.resize_images(tf.image.resize_image_with_crop_or_pad(image, min_size, min_size), [output_height, output_width])

  # augmentations
  if is_training:
    # color augmentations
    image   = tf.image.random_brightness(image, max_delta=63, seed=seed1)
    cropped = tf.image.random_brightness(cropped, max_delta=63, seed=seed1)
    # image   = tf.image.random_saturation(image, lower=0.5, upper=1.5, seed=seed2)
    # cropped = tf.image.random_saturation(cropped, lower=0.5, upper=1.5, seed=seed2)
    # image   = tf.image.random_hue(image, max_delta=0.2, seed=seed3)
    # cropped = tf.image.random_hue(cropped, max_delta=0.2, seed=seed3)
    image   = tf.image.random_contrast(image, lower=0.2, upper=1.8, seed=seed4)
    cropped = tf.image.random_contrast(cropped, lower=0.2, upper=1.8, seed=seed4)
  #
    image = gaussian_noise_layer(image, .2)
    cropped = gaussian_noise_layer(cropped, .2)
  #
    # spatial augments
    tx = tf.random_normal(shape=[],mean=0.0, stddev=20.0,dtype=tf.float32, seed=seed2) # 正規分布的なランダム値
    ty = tf.random_normal(shape=[],mean=0.0, stddev=20.0,dtype=tf.float32, seed=seed3)
    image = tf_image_translate(image, tx=tx, ty=ty)
    cropped = tf_image_translate(cropped, tx=tx, ty=ty)
  #
  #   degrees = tf.random_normal(shape=[],mean=0.0, stddev=5.0,dtype=tf.float32, seed=seed5)
  #   image = tf.contrib.image.rotate(image, degrees * math.pi / 180, interpolation='BILINEAR')
  #   cropped = tf.contrib.image.rotate(cropped, degrees * math.pi / 180, interpolation='BILINEAR')



  # image = tf.squeeze(tf.nn.lrn(image[tf.newaxis, :], 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75), [0])
  # cropped = tf.squeeze(tf.nn.lrn(cropped[tf.newaxis, :], 2, bias=1.0, alpha=0.001 / 9.0, beta=0.75), [0])

  # tf.nn.local_response_normalization(
  #     input,
  #     depth_radius=5,
  #     bias=1,
  #     alpha=1,
  #     beta=0.5,
  #     name=None
  # )

  # Subtract off the mean and divide by the variance of the pixels.
  # image = tf.image.per_image_standardization(image)
  # cropped = tf.image.per_image_standardization(cropped)

  # #  visualize on tensorboard
  # tf.summary.image('image', image[tf.newaxis, :], 1)
  # tf.summary.image('crop', cropped[tf.newaxis, :], 1)

  # normalize -1~1
  image = tf.to_float(image)
  image = tf.subtract(image, 128.0)
  image = tf.divide(image, 128.0)
  cropped = tf.to_float(cropped)
  cropped = tf.subtract(cropped, 128.0)
  cropped = tf.divide(cropped, 128.0)
  return image, cropped

  # # cropping
  # def map(fn, arrays, dtype=tf.float32):  # n入力のmap
  #   # assumes all arrays have same leading dim
  #   indices = tf.range(tf.shape(arrays[0])[0])
  #   out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
  #   return out
  #
  # # imagesからbboxes(bboxが複数ある場合ひとつめのbbox)の範囲でcropし，batchにして返す．
  # fn = lambda image, bbox: tf.squeeze(
  #   tf.image.crop_and_resize(image[tf.newaxis, :], bbox, box_ind=[0], crop_size=[224, 224]), [0])
  # crops = map(fn, [images, bboxes])

  # image = tf.image.resize_image_with_crop_or_pad(
  #     image, output_width, output_height)

