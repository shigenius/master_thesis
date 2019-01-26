# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, letter_box_pos_to_original_pos, convert_to_original_size, _iou

import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16, vgg_arg_scope
import cv2
import copy
import config as cfg
from pathlib import Path
import csv
import os
from eval_yolo import evaluate

form ..model.py import shigenet, shigenet2, shigenet3

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


def get_annotation(image_path, txtname="subwindow_log.txt"):
    img_p = Path(image_path)
    img_obj_name = img_p.parents[1].name
    cropped_dir_p = Path(str(img_p.parent)+'_cropped')
    log_p = cropped_dir_p/txtname
    assert log_p.exists(), 'Does not exist :{0}'.format(str(log_p))

    img_id = int(img_p.stem.split('_')[1])# フレーム番号

    anno = None
    with open(str(log_p), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if int(row[0]) == img_id:
                anno = row
                break

    return anno # [frame, center_x, center_y, size_x, size_y]

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )


    classes = load_coco_names(cfg.CLASS_NAME)

    if cfg.FROZEN_MODEL:
        pass
    #
    #     t0 = time.time()
    #     frozenGraph = load_graph(cfg.FROZEN_MODEL)
    #     print("Loaded graph in {:.2f}s".format(time.time()-t0))
    #
    #     boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)
    #
    #     with tf.Session(graph=frozenGraph, config=config) as sess:
    #         t0 = time.time()
    #         detected_boxes = sess.run(
    #             boxes, feed_dict={inputs: [img_resized]})

    else:
        if cfg.TINY:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), cfg.IMAGE_SIZE, cfg.DATA_FORMAT)
        # boxes : coordinates of top left and bottom right points.
        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        #
        # for specific object recognition
        #

        vgg16_image_size = vgg_16.default_image_size

        s_class_names = cfg.S_CLASS_PATH
        s_classes = [l.split(" ") for l in open(s_class_names, "r")]
        if len(s_classes[0]):  # classフォーマットが "id classname"の場合
            s_labels = {int(l[0]): l[1].replace("\n", "") for l in s_classes}
        else:  # classフォーマットが "classname"のみの場合
            s_labels = {i: l.replace("\n", "") for i, l in enumerate(s_classes)}

        num_classes_s = len(s_labels.keys())

        num_classes_extractor = cfg.S_EXTRACTOR_NUM_OF_CLASSES
        s_model = cfg.S_CKPT_FILE

        # tensorflow pipelines op


        # specific_pred, [cropped_images_placeholder, original_images_placeholder, keep_prob, is_training] = specific_object_recognition(vgg16_image_size, num_classes_s, num_classes_extractor, extractor_name)

        variables_to_restore = slim.get_variables_to_restore(include=["shigeNet_v1"])
        restorer = tf.train.Saver(variables_to_restore)
        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, cfg.CKPT_FILE)
            print('YOLO v3 Model restored in {:.2f}s'.format(time.time()-t0), "from:", cfg.CKPT_FILE)

            t0 = time.time()
            restorer.restore(sess, s_model)
            print('Specific object recognition Model restored in {:.2f}s'.format(time.time() - t0), "from:", s_model)

            # prepare test set
            with open(cfg.TEST_FILE_PATH, 'r') as f:
                f_ = [line.rstrip().split() for line in f]

            data = [[l, get_annotation(l[0], txtname=cfg.GT_INFO_FILE_NAME)] for l in f_]  # data: [[(path_str, label), [frame, center_x, center_y, size_x, size_y]],...]
            data = [l for l in data if l[1] is not None]  # annotationを取得できなかった画像は飛ばす

            def is_cropped_file_Exist(orig_filepath):
                d, file =  os.path.split(orig_filepath)
                cropped_d = d+"_cropped"
                cropped_file = os.path.join(cropped_d, file)
                return os.path.exists(cropped_file)

            data = [l for l in data if is_cropped_file_Exist(l[0][0])] # 対となるcrop画像がない画像は飛ばす


            # log
            f = open(cfg.OUTPUT_LOG_PATH, 'w')
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(
                ['image path', 'class/movie_name', 'IoU', 'Average Precision', 'TP', 'FP', 'FN', 'is RoI detected?',
                 'gt label', ' highest_conf_label', 'detect time', 'recog time'])

            total_iou = [] # 画像毎のiouのリスト
            total_ap = []# 画像毎のaverage precisionのリスト
            total_tp = 0
            total_fp = 0
            total_fn = 0

            # iterative run
            for count, gt in enumerate(data):  # gt: [(path_str, label), [frame, center_x, center_y, size_x, size_y]
                # for evaluation
                gt_box = [float(i) for i in gt[1][1:]]
                gt_box = [gt_box[0] - (gt_box[2] / 2), gt_box[1] - (gt_box[3] / 2), gt_box[0] + (gt_box[2] / 2),
                          gt_box[1] + (gt_box[3] / 2)]
                gt_label = int(gt[0][1])
                gt_anno = {gt_label: gt_box}

                print(count, ":", gt[0][0])
                img = Image.open(gt[0][0])
                img_resized = letter_box_image(img, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 128)
                img_resized = img_resized.astype(np.float32)

                t0 = time.time()
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})

                filtered_boxes = non_max_suppression(detected_boxes,
                                                     confidence_threshold=cfg.CONF_THRESHOLD,
                                                     iou_threshold=cfg.IOU_THRESHOLD)
                detect_time = time.time()-t0

                print("detected boxes in :{:.2f}s ".format(detect_time), filtered_boxes)


                # specific object recognition!
                np_img = np.array(img) / 255
                target_label = 0 # seesaaの場合 (データセットのクラス番号毎にここを変える．)


                if len(filtered_boxes.keys()) != 0: # 何かしら検出された時
                    is_detected = True

                    # get specific object name
                    for cls, bboxs in filtered_boxes.items():
                        if cls == target_label: # ターゲットラベルなら
                            print("target class detected!")
                            cropped_boxes = []
                            bboxs_ = copy.deepcopy(bboxs) # convert_to_original_size()がbboxを破壊してしまうため
                            for box, score in bboxs:
                                orig_size_box = convert_to_original_size(box, np.array((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), np.array(img.size), True)
                                # print(orig_size_box)
                                orig_size_cropped_image = np_img[int(orig_size_box[1]):int(orig_size_box[3]), int(orig_size_box[0]):int(orig_size_box[2])]
                                cropped_boxes.append(orig_size_cropped_image)

                                # cv2.imshow('result', cropped_image)
                                # cv2.waitKey(0)

                            cropped_images = []
                            for bbox in cropped_boxes:
                                cropped_images.append(cv2.resize(padding(bbox), (vgg16_image_size, vgg16_image_size)))

                            input_original = cv2.resize(padding(np_img), (vgg16_image_size, vgg16_image_size))
                            input_original = np.tile(input_original, (len(cropped_boxes), 1, 1, 1)) # croppedと同じ枚数分画像を重ねる

                            input_cropped = np.asarray(cropped_images)

                            t0 = time.time()
                            pred = sess.run(specific_pred, feed_dict={cropped_images_placeholder: input_cropped,
                                                                   original_images_placeholder: input_original,
                                                                   keep_prob: 1.0,
                                                                   is_training: False})


                            recog_time = time.time() - t0
                            print("Predictions found in {:.2f}s".format(recog_time))

                            # pred_label = [s_labels[i] for i in pred.tolist()] # idからクラス名を得る

                            classes = [s_labels[i] for i in range(num_classes_s)]

                            filtered_boxes = {}
                            for i, n in enumerate(pred.tolist()):
                                if n in filtered_boxes.keys():
                                    filtered_boxes[n].extend([bboxs_[i]]) # filtered box
                                else:
                                    filtered_boxes[n] = [bboxs_[i]]


                    # evaluation
                    print("specific obj:", filtered_boxes)
                    [tp, fp, fn], iou, ap, highest_conf_label = evaluate(filtered_boxes, gt_anno, img,
                                                                         thresh=0.1)  # 一枚の画像の評価を行う

                else:#何も検出されなかった時
                    is_detected = False
                    iou = 0.0
                    ap = 0.0
                    tp = 0
                    fp = 0
                    fn = len(gt_anno.values())
                    highest_conf_label = -1

                total_iou.append(iou)
                total_ap.append(ap)
                print("IoU:", iou)
                print("average Precision:", ap)
                print("mean average IoU:", sum(total_iou) / (len(total_iou) + 1e-05))
                print("mean Average Precision:", sum(total_ap) / (len(total_ap) + 1e-05))

                total_tp += tp
                total_fp += fp
                total_fn += fn

                # draw pred_bbox
                draw_boxes(filtered_boxes, img, classes, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), True)
                # draw GT
                draw = ImageDraw.Draw(img)
                color = (0, 0, 0)
                draw.rectangle(gt_box, outline=color)
                draw.text(gt_box[:2], 'GT_' + classes[gt_label], fill=color)

                img.save(os.path.join(cfg.OUTPUT_DIR, '{0:04d}_'.format(count) + os.path.basename(gt[0][0])))

                movie_name = os.path.basename(os.path.dirname(gt[0][0]))
                movie_parant_dir = os.path.basename(os.path.dirname(os.path.dirname(gt[0][0])))
                pred_label = classes[highest_conf_label] if highest_conf_label != -1 else "None"
                writer.writerow([gt[0][0], os.path.join(movie_name, movie_parant_dir), iou, ap, tp, fp, fn, is_detected,
                                 classes[gt_label], pred_label, detect_time, recog_time])

            print("total tp :", total_tp)
            print("total fp :", total_fp)
            print("total fn :", total_fn)
            f.close()
            print("proc finished.")


if __name__ == '__main__':
    tf.app.run()
