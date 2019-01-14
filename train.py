import tensorflow as tf

from datasets import shisa_instances
from model import shigenet, load_batch

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/',
                    'Directory with the data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', 10000,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS

def main(args):
    # load the dataset
    dataset = shisa_instances.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, crops, labels, bboxes = load_batch(
        dataset,
        FLAGS.batch_size,
        height=shigenet.default_input_size,
        width=shigenet.default_input_size,
        is_training=True)

    # run the image through the model
    predictions = shigenet(images, crops, dataset.num_classes)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    # for logging
    accuracy = slim.metrics.accuracy(
        tf.argmax(predictions, 1),
        tf.argmax(one_hot_labels, 1))  # ... or whatever metrics needed

    def train_step_fn(session, *args, **kwargs): # custom train_step_fn
        # 1回の勾配計算を実行するために呼び出す関数
        # 4つの引数（現在のセッション、学習処理、グローバルトレーニングのステップ、キーワード引数の辞書）が必要です
        # defaultのtrain_step_fnを使う場合はslim.learning.train_step_fnを指定する．
        total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)

        # added
        if train_step_fn.step % 1 == 0:
            acc = session.run(train_step_fn.accuracy)
            # filenames = session.run(train_step_fn.fnames)
            print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (
            str(train_step_fn.step).rjust(6, '0'), total_loss, acc * 100))
            # print("filename %s" % filenames)

        train_step_fn.step += 1
        return [total_loss, should_stop]

    train_step_fn.step = 0
    train_step_fn.accuracy = accuracy
    # train_step_fn.fnames = fnames

    init_op = tf.global_variables_initializer()

    # Restore only the convolutional layers:
    # variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
    # print(variables_to_restore)
    # init_fn = tf.assign_from_checkpoint_fn(model_path, variables_to_restore)

    # restorer = tf.train.Saver()
    # tf.global_variables_initializer()


    print("Run training")
    # run training
    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20,
        init_op=init_op,
        train_step_fn=train_step_fn)

if __name__ == '__main__':
  tf.app.run()