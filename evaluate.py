import tensorflow as tf

from datasets import shisa_instances
from model import shigenet, load_batch

slim = tf.contrib.slim
metrics = tf.contrib.metrics

import os
import logging
logging.basicConfig(level=logging.DEBUG)

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/',
                    'Directory with the data.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory with the log data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
flags.DEFINE_string('checkpoint_name', 'model.ckpt-134052',
                    '')
flags.DEFINE_integer('eval_interval_secs', 1,
                    'Number of seconds between evaluations.')
FLAGS = flags.FLAGS

def main(args):
    ckpt_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoint_name)

    # load the dataset
    dataset = shisa_instances.get_split('train', FLAGS.data_dir)
    # load batch of dataset
    images, crops, labels, bboxes, fnames, vnames = load_batch(
        dataset,
        FLAGS.batch_size,
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

    initial_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

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

        print(names_to_values)
        for batch in range(20000):
            values = sess.run([accuracy, labels, predictions1D, fnames, vnames])
            print(values)

    # metric_values = slim.evaluation.evaluate_once(
    #     '',
    #     ckpt_path,
    #     FLAGS.log_dir,
    #     num_evals=10,
    #     initial_op=initial_op,
    #     eval_op=list(names_to_updates.values()),
    #     final_op=list(names_to_values.values()))

    # print(metric_values)


if __name__ == '__main__':
  tf.app.run()