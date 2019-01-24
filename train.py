import tensorflow as tf

from datasets import shisa_instances
from model import shigenet, load_batch

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/',
                    'Directory with the data.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('num_batches', 50000,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
flags.DEFINE_string('extractor_ckpt', '/Users/shigetomi/Downloads/vgg_16.ckpt',
                    '')
flags.DEFINE_integer('val_freq', 1000, 'validation freq per step')
flags.DEFINE_integer('val_num_batch', 20, 'num of running validation per step')

FLAGS = flags.FLAGS

def main(args):

    # load the dataset
    train_dataset = shisa_instances.get_split('train', FLAGS.data_dir)
    valid_dataset = shisa_instances.get_split('val', FLAGS.data_dir)
    # load batch of dataset
    train_images, train_crops, train_labels, train_bboxes, train_fnames, train_vnames = load_batch(
        train_dataset,
        FLAGS.batch_size,
        height=shigenet.default_input_size,
        width=shigenet.default_input_size,
        is_training=True)
    valid_images, valid_crops, valid_labels, valid_bboxes, valid_fnames, valid_vnames = load_batch(
        valid_dataset,
        FLAGS.batch_size,
        height=shigenet.default_input_size,
        width=shigenet.default_input_size,
        is_training=False)

    # run the image through the model
    train_predictions = shigenet(train_images, train_crops, train_dataset.num_classes, is_training=True, reuse=None)
    valid_predictions = shigenet(valid_images, valid_crops, valid_dataset.num_classes, is_training=False, reuse=True)

    # get the cross-entropy loss
    train_one_hot_labels = slim.one_hot_encoding(
        train_labels,
        train_dataset.num_classes)
    valid_one_hot_labels = slim.one_hot_encoding(
        valid_labels,
        valid_dataset.num_classes)

    slim.losses.softmax_cross_entropy(
        train_predictions,
        train_one_hot_labels)

    # define on  previous calc valid loss
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('train_total_loss', total_loss)

    valid_loss = slim.losses.softmax_cross_entropy(
        valid_predictions,
        valid_one_hot_labels)
    tf.summary.scalar('valid_loss', valid_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    # for logging
    train_accuracy = slim.metrics.accuracy(
        tf.argmax(train_predictions, 1),
        tf.argmax(train_one_hot_labels, 1))  # ... or whatever metrics needed
    valid_accuracy = slim.metrics.accuracy(
        tf.argmax(valid_predictions, 1),
        tf.argmax(valid_one_hot_labels, 1))  # ... or whatever metrics needed
    tf.summary.scalar('train_acc', train_accuracy)
    tf.summary.scalar('valid_acc', valid_accuracy)

    def train_step_fn(session, *args, **kwargs): # custom train_step_fn
        # 1回の勾配計算を実行するために呼び出す関数
        # 4つの引数（現在のセッション、学習処理、グローバルトレーニングのステップ、キーワード引数の辞書）が必要です
        # defaultのtrain_step_fnを使う場合はslim.learning.train_step_fnを指定する．
        total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)

        # get acc and loss for logging
        if train_step_fn.step % 1 == 0:
            train_acc = session.run(train_step_fn.train_accuracy)
            # filenames = session.run(train_step_fn.fnames)
            print('Step %s - Loss: %.4f Accuracy: %.2f%%' % (
            str(train_step_fn.step).rjust(6, '0'), total_loss, train_acc * 100))
            # print("filename %s" % filenames)

        # validation
        if train_step_fn.step % FLAGS.val_freq == 0:

            val_acc_l = []
            val_loss_l = []
            # filenames_l = []
            # vnames_l = []
            for i in range(FLAGS.val_num_batch):
                # val_acc, val_loss, filenames, vnames = session.run([train_step_fn.valid_accuracy, valid_loss, train_step_fn.fnames, train_step_fn.vnames])
                val_acc, val_loss = session.run(
                    [train_step_fn.valid_accuracy, valid_loss])
                val_acc_l.append(val_acc)
                val_loss_l.append(val_loss)
                # filenames_l.append(filenames)
                # vnames_l.append(vnames)

            ave_val_acc = sum(val_acc_l) / len(val_acc_l)
            ave_val_loss = sum(val_loss_l) / len(val_loss_l)
            print('Step %s - Average(item:%d) Validation Loss: %.2f Accuracy: %.2f%%' % (
            str(train_step_fn.step).rjust(6, '0'), FLAGS.val_num_batch*FLAGS.batch_size, ave_val_loss, ave_val_acc * 100))
            # print("filename %s" % filenames_l)
            # print("video_name %s" % vnames_l)

        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.train_accuracy = train_accuracy
    train_step_fn.valid_accuracy = valid_accuracy
    # train_step_fn.fnames = valid_fnames
    # train_step_fn.vnames = valid_vnames

    init_op = tf.global_variables_initializer()

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

    print("extractor restored from:", FLAGS.extractor_ckpt)
    print("Run training")
    # run training
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20,
        save_interval_secs=60,
        init_op=init_op,
        # init_fn=init_fn,
        train_step_fn=train_step_fn,
        number_of_steps=FLAGS.num_batches)

if __name__ == '__main__':
  tf.app.run()