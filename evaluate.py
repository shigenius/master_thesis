import tensorflow as tf

from datasets import shisa_instances
from model import shigenet, load_batch


slim = tf.contrib.slim
metrics = tf.contrib.metrics

import os
from utils import label_map_util
import csv
import time
import logging
logging.basicConfig(level=logging.DEBUG)

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/',
                    'Directory with the data.')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory with the log data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
flags.DEFINE_string('checkpoint_name', 'model.ckpt-134052',
                    '')
flags.DEFINE_string('labelfile_name', 'pascal_label_map.pbtxt',
                    'label file name that discribed pascal voc')
flags.DEFINE_string('eval_log_name', 'eval.csv',
                    'output log name')
flags.DEFINE_integer('eval_interval_secs', 1,
                    'Number of seconds between evaluations.')
FLAGS = flags.FLAGS

def main(args):
    label_name_to_id = label_map_util.get_label_map_dict(os.path.join(FLAGS.data_dir, FLAGS.labelfile_name))
    label_id_to_name = {items[1]:items[0] for items in label_name_to_id.items()}
    # eval log
    f = open(os.path.join(FLAGS.log_dir, FLAGS.eval_log_name), 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['file_path', 'source_video', 'gt',  'prediction', 'is_correct_label', 'running_time',])

    ckpt_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint_name)

    # load the dataset
    dataset = shisa_instances.get_split('test', FLAGS.data_dir)

    # load batch of dataset
    images, crops, labels, bboxes, filenames, videonames = load_batch(
        dataset,
        1,
        height=shigenet.default_input_size,
        width=shigenet.default_input_size,
        is_training=False,
        shuffle=False)

    # run the image through the model
    predictions = shigenet(images, crops, dataset.num_classes, is_training=False, reuse=None)
    predictions1D = tf.to_int64(tf.argmax(predictions, 1))

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        "accuracy": slim.metrics.streaming_accuracy(predictions1D, labels),
        # "mse": slim.metrics.streaming_mean_squared_error(predictions, labels),
        'precision': slim.metrics.streaming_precision(predictions1D, labels)
    })
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)

    accuracy = slim.metrics.accuracy(
        tf.argmax(predictions, 1),
        tf.argmax(one_hot_labels, 1))  # ... or whatever metrics needed

    # initial_op = tf.group(
    #     tf.global_variables_initializer(),
    #     tf.local_variables_initializer())

    variables_to_restore = slim.get_model_variables()
    restorer = tf.train.Saver(variables_to_restore)

    print("run evaluating.")
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, ckpt_path)
        print("model restored from:", ckpt_path)

        # adding these 2 lines fixed the hang forever problem
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num_record = len(list(tf.python_io.tf_record_iterator(dataset.data_sources)))
        # print("num of record:", num_record)
        num_correct = 0
        for batch in range(num_record):
            start_time = time.time()
            acc, label, pred, fname, vname = sess.run([accuracy, labels, predictions1D, filenames, videonames])
            elapsed_time = time.time() - start_time

            label = label_id_to_name[label.tolist()[0]]
            pred = label_id_to_name[pred.tolist()[0]]
            print(batch, acc, fname, vname, label, pred)
            is_correct = True if label == pred else False
            writer.writerow([fname[0].decode('utf-8'), label+"/"+vname[0].decode('utf-8'), label, pred, is_correct, elapsed_time])

            if is_correct is True:
                num_correct += 1

        print("num of record:", num_record, "num of correct:", num_correct, "Acc:", num_correct/num_record)
    print("process finished")
    f.close()



if __name__ == '__main__':
  tf.app.run()