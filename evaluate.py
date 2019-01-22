import tensorflow as tf

from datasets import shisa_instances
from model import shigenet, load_batch

slim = tf.contrib.slim
metrics = tf.contrib.metrics

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/',
                    'Directory with the data.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('num_evals', 10000,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/eval',
                    'Directory with the log data.')
flags.DEFINE_string('checkpoint_dir', './log/train',
                    'Directory with the model checkpoint data.')
flags.DEFINE_integer('eval_interval_secs', 1,
                    'Number of seconds between evaluations.')
FLAGS = flags.FLAGS

def main(args):

    # load the dataset
    dataset = shisa_instances.get_split('test', FLAGS.data_dir)
    # load batch of dataset
    images, crops, labels, bboxes, fnames, vnames = load_batch(
        dataset,
        FLAGS.batch_size,
        height=shigenet.default_input_size,
        width=shigenet.default_input_size,
        is_training=False)

    # run the image through the model
    predictions = shigenet(images, crops, dataset.num_classes, is_training=False, reuse=None)
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    # Choose the metrics to compute:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
        "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
    })
    print("run evaluating.")
    # Evaluate the model using 1000 batches of data:
    num_batches = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for batch_id in range(num_batches):
            print(batch_id)
            hoge = sess.run(list(names_to_updates.values()))
            print(hoge)

        metric_values = sess.run(names_to_values.values())
        for metric, value in zip(names_to_values.keys(), metric_values):
            print('Metric %s has value: %f' % (metric, value))

if __name__ == '__main__':
  tf.app.run()