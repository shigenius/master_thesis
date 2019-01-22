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

    # convert prediction values for each class into single class prediction
    # predictions = tf.to_int64(tf.argmax(predictions, 1))
    predictions = tf.to_int64(tf.argmax(predictions, 1))

    # streaming metrics to evaluate
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    #     "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
    #     # 'accuracy': slim.metrics.accuracy(predictions=predictions, labels=labels),
    #     # 'precision': tf.metrics.precision(predictions, labels),
    #     # 'recall': tf.metrics.recall(predictions, labels)
    # })
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'accuracy': slim.metrics.accuracy(predictions, labels),
    #     'precision': slim.metrics.precision(predictions, labels),
    #     'recall': slim.metrics.recall(mean_relative_errors, 0.3),
    # })

    # streaming metrics to evaluate
    metrics_to_values, metrics_to_updates = metrics.aggregate_metric_map({
        'eval/mse': metrics.streaming_mean_squared_error(predictions, labels),
        'eval/accuracy': metrics.streaming_accuracy(predictions, labels),
    })

    # write the metrics as summaries
    # summary_ops = []
    # for metric_name, metric_value in names_to_values.items():
    #     op = tf.summary.scalar(metric_name, metric_value)
    #     op = tf.Print(op, [metric_value], metric_name)
    #     summary_ops.append(op)

    for metric_name, metric_value in metrics_to_values.items():
        tf.summary.scalar(metric_name, metric_value)

    # evaluate on the model saved at the checkpoint directory
    # evaluate every eval_interval_secs

    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # checkpoint_path = "/Users/shigetomi/dev/master_thesis/log/train/model.ckpt-132509"
    # print_tensors_in_checkpoint_file(file_name=checkpoint_path, all_tensors=False, tensor_name='', all_tensor_names=True)

    print("run evaluating.")
    # slim.evaluation.evaluation_loop(
    #     '',
    #     FLAGS.checkpoint_dir,
    #     FLAGS.log_dir,
    #     num_evals=FLAGS.num_batches,
    #     eval_op=list(names_to_updates.values()),
    #     summary_op=tf.summary.merge(summary_ops),
    #     eval_interval_secs=FLAGS.eval_interval_secs)

    slim.evaluation.evaluation_loop(
        '',
        FLAGS.checkpoint_dir,
        FLAGS.log_dir,
        num_evals=FLAGS.num_evals,
        eval_op=list(metrics_to_updates.values()),
        eval_interval_secs=FLAGS.eval_interval_secs)
    # def train_step_fn(session, *args, **kwargs): # custom train_step_fn
    #     # 1回の勾配計算を実行するために呼び出す関数
    #     # 4つの引数（現在のセッション、学習処理、グローバルトレーニングのステップ、キーワード引数の辞書）が必要です
    #     # defaultのtrain_step_fnを使う場合はslim.learning.train_step_fnを指定する．
    #     total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)
    #
    #     # get acc and loss for logging
    #     if train_step_fn.step % 1 == 0:
    #         train_acc = session.run(train_step_fn.train_accuracy)
    #         # filenames = session.run(train_step_fn.fnames)
    #         print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (
    #         str(train_step_fn.step).rjust(6, '0'), total_loss, train_acc * 100))
    #         # print("filename %s" % filenames)
    #
    #     # validation
    #     if train_step_fn.step % FLAGS.val_freq == 0:
    #
    #         val_acc_l = []
    #         val_loss_l = []
    #         # filenames_l = []
    #         for i in range(FLAGS.val_num_batch):
    #             # val_acc, val_loss, filenames = session.run([train_step_fn.valid_accuracy, valid_loss, train_step_fn.fnames])
    #             val_acc, val_loss = session.run(
    #                 [train_step_fn.valid_accuracy, valid_loss])
    #             val_acc_l.append(val_acc)
    #             val_loss_l.append(val_loss)
    #             # filenames_l.append(filenames)
    #
    #         ave_val_acc = sum(val_acc_l) / len(val_acc_l)
    #         ave_val_loss = sum(val_loss_l) / len(val_loss_l)
    #         print('Step %s - Average(item:%d) Validation Loss: %.2f Accuracy: %.2f%%' % (
    #         str(train_step_fn.step).rjust(6, '0'), FLAGS.val_num_batch*FLAGS.batch_size, ave_val_loss, ave_val_acc * 100))
    #         # print("filename %s" % filenames_l)
    #
    #     train_step_fn.step += 1
    #     return [total_loss, should_stop]
    #
    # train_step_fn.step = 0
    # train_step_fn.train_accuracy = train_accuracy
    # # train_step_fn.fnames = valid_fnames
    #
    # init_op = tf.global_variables_initializer()

    # def name_in_checkpoint(var):
    #     if "shigenet" in var.op.name:
    #         return var.op.name.replace("shigenet/extractor/", "")
    #
    # # Restore only the convolutional layers:
    # variables_to_restore = slim.get_variables_to_restore(include=['shigenet', 'extractor'], exclude=['vgg16/fc6', 'vgg16/fc7', 'vgg16/fc8'])
    # variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore if
    #                         "vgg_16" in var.op.name and 'RMSProp' not in var.op.name}
    # print(variables_to_restore)
    # init_fn = slim.assign_from_checkpoint_fn(FLAGS.extractor_ckpt, variables_to_restore)

    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # print_tensors_in_checkpoint_file(FLAGS.extractor_ckpt, all_tensors=True, tensor_name='', all_tensor_names="")
    # restorer = tf.train.Saver()
    # tf.global_variables_initializer()

    # run training
    # slim.learning.train(
    #     train_op,
    #     FLAGS.log_dir,
    #     save_summaries_secs=20,
    #     init_op=init_op,
    #     # init_fn=init_fn,
    #     train_step_fn=train_step_fn)

if __name__ == '__main__':
  tf.app.run()