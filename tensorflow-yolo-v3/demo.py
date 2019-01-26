# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, letter_box_pos_to_original_pos, convert_to_original_size

import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16, vgg_arg_scope
import cv2
import copy

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.9, 'Gpu memory fraction to use')

# added
tf.app.flags.DEFINE_string(
    's_model', '', 'path of pre-trained specific object recognition model (CBO-Net)')
# tf.app.flags.DEFINE_string(
#     'extractor_model', '', 'path of pre-trained general recognition model (VGG_16)')
tf.app.flags.DEFINE_string(
    's_class_names', '', 'File with specific object class names')
tf.app.flags.DEFINE_integer(
    'num_classes_g', 1, 'num of classes for general object recognition')

def shigeNet_v1(cropped_images, original_images, num_classes_s, num_classes_g, keep_prob=1.0, is_training=True, scope='shigeNet_v1', reuse=None, extractor_name='vgg_16'):
    end_points = {}
    with tf.variable_scope(scope, 'shigeNet_v1', reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # Extract features
            with slim.arg_scope(vgg_arg_scope()):
                logits_c, end_points_c = vgg_16(cropped_images, num_classes=num_classes_g, is_training=False, reuse=None)
                logits_o, end_points_o = vgg_16(original_images, num_classes=num_classes_g, is_training=False, reuse=True)

                feature_c = end_points_c['shigeNet_v1/vgg_16/fc7']
                feature_o = end_points_o['shigeNet_v1/vgg_16/fc7']

                # feature map summary
                # Tensorを[-1,7,7,ch]から[-1,ch,7,7]と順列変換し、[-1]と[ch]をマージしてimage出力
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_c', tf.reshape(tf.transpose(end_points_c['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)
                tf.summary.image('shigeNet_v1/vgg_16/conv5/conv5_3_o', tf.reshape(tf.transpose(end_points_o['shigeNet_v1/vgg_16/conv5/conv5_3'], perm=[0, 3, 1, 2]), [-1, 14, 14, 1]), 10)

            # Concat!
            with tf.variable_scope('Concat') as scope:
                concated_feature = tf.concat([tf.layers.Flatten()(feature_c), tf.layers.Flatten()(feature_o)], 1)  # (?, x, y, z)

            with tf.variable_scope('Logits'):
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    weights_regularizer=slim.l2_regularizer(0.0005)):
                    net = slim.fully_connected(concated_feature, 1000, scope='fc1')
                    net = slim.dropout(net, keep_prob, scope='dropout1')
                    net = slim.fully_connected(net, 256, scope='fc2')
                    net = slim.dropout(net, keep_prob, scope='dropout2')
                    net = slim.fully_connected(net, num_classes_s, activation_fn=None, scope='fc3')

                    end_points['Logits'] = net
                    # squeeze = tf.squeeze(net, [1, 2]) # 次元1,2の要素数が1であるならばその次元を減らす
                    # end_points['Predictions'] = tf.nn.softmax(squeeze, name='Predictions')
                    end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

        return end_points


def padding(image):
    # アス比の違う画像をゼロパディングして正方形にする
    w = image.shape[1]
    h = image.shape[0]
    if w == h:
        return image
    elif w > h:
        offset = w - h
        n = int(offset / 2)
        if offset % 2 == 0:
            dst = np.pad(image, [(n, n), (0, 0), (0, 0)], 'constant')
        else:
            dst = np.pad(image, [(n, n+1), (0, 0), (0, 0)], 'constant')
        return dst

    else:
        offset = h - w
        n = int(offset / 2)
        if offset % 2 == 0:
            dst = np.pad(image, [(0, 0), (n, n), (0, 0)], 'constant')
        else:
            dst = np.pad(image, [(0, 0), (n, n+1), (0, 0)], 'constant')
        return dst

def specific_object_recognition(image_size, num_classes_s, num_classes_g, extractor_name='vgg_16'):
    # specific object recognition!
    with tf.name_scope('input'):
        with tf.name_scope('cropped_images'):
            cropped_images_placeholder = tf.placeholder(dtype="float32",
                                                        shape=(None, image_size, image_size, 3))
        with tf.name_scope('original_images'):
            original_images_placeholder = tf.placeholder(dtype="float32",
                                                         shape=(None, image_size, image_size, 3))
        with tf.name_scope('labels'):
            labels_placeholder = tf.placeholder(dtype="float32", shape=(None, num_classes_s))
        keep_prob = tf.placeholder(dtype="float32")
        is_training = tf.placeholder(dtype="bool")  # train flag

    # Build the graph
    end_points = shigeNet_v1(cropped_images=cropped_images_placeholder, original_images=original_images_placeholder,
                             extractor_name=extractor_name, num_classes_s=num_classes_s, num_classes_g=num_classes_g,
                             is_training=is_training, keep_prob=keep_prob)
    logits = end_points["Logits"]
    predictions = end_points["Predictions"]
    predict_labels = tf.argmax(predictions, 1)

    return predict_labels, [cropped_images_placeholder, original_images_placeholder, keep_prob, is_training]

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    img = Image.open(FLAGS.input_img)
    img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.Session(graph=frozenGraph, config=config) as sess:
            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)
        # boxes : coordinates of top left and bottom right points.
        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, FLAGS.ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)

    print("detected boxes:", filtered_boxes)


    # specific object recognition!
    np_img = np.array(img) / 255
    target_label = 0 # seesaaの場合 (データセットのクラス番号毎にここを変える．)


    for cls, bboxs in filtered_boxes.items():
        if cls == target_label: # ターゲットラベルなら
            print("target class detected!")
            bounding_boxes = []
            bboxs_ = copy.deepcopy(bboxs) # convert_to_original_size()がbboxを破壊してしまうため
            for box, score in bboxs:
                orig_size_box = convert_to_original_size(box, np.array((FLAGS.size, FLAGS.size)), np.array(img.size), True)
                print(orig_size_box)
                cropped_image = np_img[int(orig_size_box[1]):int(orig_size_box[3]), int(orig_size_box[0]):int(orig_size_box[2])]
                bounding_boxes.append(cropped_image)

                # cv2.imshow('result', cropped_image)
                # cv2.waitKey(0)

            vgg16_image_size = vgg_16.default_image_size

            s_class_names = FLAGS.s_class_names
            s_classes = [l.split(" ") for l in open(s_class_names, "r")]
            if len(s_classes[0]):  # classフォーマットが "id classname"の場合
                s_labels = {int(l[0]): l[1].replace("\n", "") for l in s_classes}
            else:  # classフォーマットが "classname"のみの場合
                s_labels = {i: l.replace("\n", "") for i, l in enumerate(s_classes)}

            num_classes_s = len(s_labels.keys())

            num_classes_g = FLAGS.num_classes_g
            # model_path = FLAGS.extractor_model
            s_model = FLAGS.s_model

            extractor_name = 'vgg_16'

            specific_pred, [cropped_images_placeholder, original_images_placeholder, keep_prob, is_training] = specific_object_recognition(vgg16_image_size, num_classes_s, num_classes_g, extractor_name)

            variables_to_restore = slim.get_variables_to_restore(include=["shigeNet_v1"])
            restorer = tf.train.Saver(variables_to_restore)
            # print(variables_to_restore)

            with tf.Session(config=config) as sess:
                # ext_restorer.restore(sess, model_path)
                # print("Extractor Model restored from:", model_path)
                t0 = time.time()
                restorer.restore(sess, s_model)
                print('Specific object recognition Model restored in {:.2f}s'.format(time.time() - t0), "from:", s_model)

                input_original = cv2.resize(padding(np_img), (vgg16_image_size, vgg16_image_size))
                input_original = np.tile(input_original, (len(bounding_boxes), 1, 1, 1)) # croppedと同じ枚数分画像を重ねる

                cropped_images= []
                for bbox in bounding_boxes:
                    cropped_images.append(cv2.resize(padding(bbox), (vgg16_image_size, vgg16_image_size)))

                input_cropped = np.asarray(cropped_images)

                t0 = time.time()
                pred = sess.run(specific_pred, feed_dict={cropped_images_placeholder: input_cropped,
                                                       original_images_placeholder: input_original,
                                                       keep_prob: 1.0,
                                                       is_training: False})

                print("Predictions found in {:.2f}s".format(time.time() - t0))

                # pred_label = [s_labels[i] for i in pred.tolist()]

                classes = [s_labels[i] for i in range(num_classes_s)]

                filtered_boxes = {}
                for i, n in enumerate(pred.tolist()):
                    if n in filtered_boxes.keys():
                        filtered_boxes[n].extend([bboxs_[i]])
                    else:
                        filtered_boxes[n] = [bboxs_[i]]


    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

    img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
