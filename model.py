import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

vgg = nets.vgg

from preprocessing import lenet_preprocessing

slim = tf.contrib.slim
import datasets.shisa_instances

def lenet(images):
    net = slim.conv2d(images, 20, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 50, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
    return net


def shigenet(images, crops, num_classes, dropout=0.5, is_training=False, reuse=None):
    with tf.variable_scope('shigenet', reuse=reuse) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with tf.variable_scope('branch_crop') as scope:
                net_l = slim.conv2d(crops, 20, [3, 3], padding='VALID', scope='conv1')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool1')
                net_l = slim.conv2d(net_l, 50, [3, 3], padding='VALID', scope='conv2')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool2')
                net_l = slim.conv2d(net_l, 100, [3, 3], padding='VALID', scope='conv3')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool3')
                net_l = slim.conv2d(net_l, 200, [3, 3], padding='VALID', scope='conv4')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool4')
                net_l = slim.conv2d(net_l, 500, [3, 3], padding='VALID', scope='conv5')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool5')
                net_l = slim.conv2d(net_l, 1024, [3, 3], padding='VALID', scope='conv6')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool6')
                net_l = slim.conv2d(net_l, 1024, [1, 1], padding='VALID', scope='pw-conv')
                print(net_l)

            with tf.variable_scope('branch_orig') as scope:
                net_g = slim.conv2d(images, 20, [3, 3], padding='VALID', scope='conv1')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool1')
                net_g = slim.conv2d(net_g, 50, [3, 3], padding='VALID', scope='conv2')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool2')
                net_g = slim.conv2d(net_g, 100, [3, 3], padding='VALID', scope='conv3')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool3')
                net_g = slim.conv2d(net_g, 200, [3, 3], padding='VALID', scope='conv4')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool4')
                net_g = slim.conv2d(net_g, 500, [3, 3], padding='VALID', scope='conv5')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool5')
                net_g = slim.conv2d(net_g, 1024, [3, 3], padding='VALID', scope='conv6')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool6')
                net_g = slim.conv2d(net_g, 1024, [1, 1], padding='VALID', scope='pw-conv')
                print(net_g)

            with tf.variable_scope('logit') as scope:
                net = tf.concat([net_l, net_g], 3)
                # net = tf.add(net_l, net_g)
                net = slim.flatten(net, scope='flatten')
                net = slim.fully_connected(net, 1000, scope='fc1')
                net = slim.dropout(net, dropout, scope='dropout1')
                net = slim.fully_connected(net, 500, scope='fc2')
                net = slim.dropout(net, dropout, scope='dropout2')
                net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc3')
                # show_variables()
        return net


def shigenet2(images, crops, num_classes, keep_prob=0.8, is_training=False, reuse=None):
    # focus branches
    with tf.variable_scope('shigenet2', reuse=reuse) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with tf.variable_scope('branch_crop') as scope:
                # net_l = shigenet3_block(crops, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_l = slim.conv2d(crops, 20, [3, 3], padding='VALID', scope='conv1')
                print(net_l)
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool1')
                print(net_l)
                net_l = slim.conv2d(net_l, 50, [3, 3], padding='VALID', scope='conv2')
                print(net_l)
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool2')
                print(net_l)
                net_l = slim.conv2d(net_l, 100, [3, 3], padding='VALID', scope='conv3')
                print(net_l)
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool3')
                print(net_l)
                net_l = slim.conv2d(net_l, 200, [3, 3], padding='VALID', scope='conv4')
                print(net_l)
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool4')
                print(net_l)
                net_l = slim.conv2d(net_l, 500, [3, 3], padding='VALID', scope='conv5')
                print(net_l)
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool5')
                print(net_l)
                net_l = slim.conv2d(net_l, 1024, [3, 3], padding='VALID', scope='conv6')
                print(net_l)
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool6')
                print(net_l)
                net_l = slim.conv2d(net_l, 1024, [1, 1], padding='VALID', scope='pw-conv')
                print(net_l)
                net_l = slim.dropout(net_l, keep_prob, is_training=is_training, scope='dropout1')
                net_l = slim.conv2d(net_l, 500, [1, 1], padding='VALID', scope='pw-conv2')
                print(net_l)
                net_l = slim.dropout(net_l, keep_prob, is_training=is_training, scope='dropout2')
                net_l = slim.conv2d(net_l, num_classes, [1, 1], padding='VALID', scope='pw-conv3')
                print(net_l)

            with tf.variable_scope('branch_orig') as scope:
                # net_g = shigenet3_block(images, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_g = slim.conv2d(images, 20, [3, 3], padding='VALID', scope='conv1')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool1')
                net_g = slim.conv2d(net_g, 50, [3, 3], padding='VALID', scope='conv2')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool2')
                net_g = slim.conv2d(net_g, 100, [3, 3], padding='VALID', scope='conv3')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool3')
                net_g = slim.conv2d(net_g, 200, [3, 3], padding='VALID', scope='conv4')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool4')
                net_g = slim.conv2d(net_g, 500, [3, 3], padding='VALID', scope='conv5')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool5')
                net_g = slim.conv2d(net_g, 1024, [3, 3], padding='VALID', scope='conv6')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool6')
                net_g = slim.conv2d(net_g, 1024, [1, 1], padding='VALID', scope='pw-conv')
                net_g = slim.dropout(net_g, keep_prob, is_training=is_training, scope='dropout1')
                net_g = slim.conv2d(net_g, 500, [1, 1], padding='VALID', scope='pw-conv2')
                net_g = slim.dropout(net_g, keep_prob, is_training=is_training, scope='dropout2')
                net_g = slim.conv2d(net_g, num_classes, [1, 1], padding='VALID', scope='pw-conv3')

            with tf.variable_scope('logit') as scope:
                net = tf.add(net_l, net_g) # element-wise add
                net = slim.flatten(net, scope='flatten')

        return net


def shigenet2_block(input, num_classes, keep_prob=0.8, is_training=False):
    # input 224x224x3 images tensor
    # return 1x1xnum_classes tensor
    with slim.arg_scope([slim.conv2d, slim.fully_connected], padding='VALID',
                        activation_fn=tf.nn.relu):
        net = slim.conv2d(input, 20, [3, 3], padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 50, [3, 3], padding='VALID', scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.conv2d(net, 100, [3, 3], padding='VALID', scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.conv2d(net, 200, [3, 3], padding='VALID', scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.conv2d(net, 500, [3, 3], padding='VALID', scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.conv2d(net, 1024, [3, 3], padding='VALID', scope='conv6')
        net = slim.max_pool2d(net, [2, 2], scope='pool6')
        net = slim.conv2d(net, 1024, [1, 1], padding='VALID', scope='pw-conv')
        net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout1')
        net = slim.conv2d(net, 500, [1, 1], padding='VALID', scope='pw-conv2')
        net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout2')
        net = slim.conv2d(net, num_classes, [1, 1], padding='VALID', scope='pw-conv3')
        return net

def shigenet3(images, crops, num_classes, keep_prob=1.0, is_training=False, reuse=None):
    # heavy network!
    with tf.variable_scope('shigenet3', reuse=reuse) as scope:


        with tf.variable_scope('branch_crop') as scope:
            net_l = shigenet3_block(crops, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)

        with tf.variable_scope('branch_orig') as scope:
            net_g = shigenet3_block(images, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)

        with tf.variable_scope('logit') as scope:
            net = tf.add(net_l, net_g) # element-wise add
            net = slim.flatten(net, scope='flatten')

        return net


def shigenet3_block(input, num_classes, keep_prob=1.0, is_training=False):
    # input 224x224x3 images tensor
    # return 1x1xnum_classes tensor
    with slim.arg_scope([slim.conv2d, slim.fully_connected], padding='VALID',
                        activation_fn=tf.nn.relu):
        net = slim.repeat(input, 2, slim.conv2d, 20, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 3, slim.conv2d, 50, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 100, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        net = slim.conv2d(net, 512, [1, 1], padding='VALID', scope='pw-conv')
        # net = slim.dropout(net, keep_prob, is_training=is_training,
        #                    scope='dropout1')
        net = slim.conv2d(net, 100, [1, 1], padding='VALID', scope='pw-conv2')
        # net = slim.dropout(net, keep_prob, is_training=is_training,
        #                    scope='dropout2')
        net = slim.conv2d(net, num_classes, [1, 1], padding='VALID', scope='pw-conv3')
        print(net)
        return net


def shigenet2_multiply(images, crops, num_classes, keep_prob=0.8, is_training=False, reuse=None):
    # focus branches
    with tf.variable_scope('shigenet2', reuse=reuse) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with tf.variable_scope('branch_crop') as scope:
                # net_l = shigenet3_block(crops, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_l = shigenet2_block(crops, num_classes, keep_prob=keep_prob, is_training=is_training)

            with tf.variable_scope('branch_orig') as scope:
                # net_g = shigenet3_block(images, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_g = shigenet2_block(images, num_classes, keep_prob=keep_prob, is_training=is_training)

            with tf.variable_scope('logit') as scope:
                net = tf.multiply(net_l, net_g) # element-wise multiply
                net = slim.flatten(net, scope='flatten')

        return net


def shigenet2_with_sigmoid(images, crops, num_classes, keep_prob=0.8, is_training=False, reuse=None):
    # focus branches
    with tf.variable_scope('shigenet2', reuse=reuse) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with tf.variable_scope('branch_crop') as scope:
                # net_l = shigenet3_block(crops, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_l = slim.conv2d(crops, 20, [3, 3], padding='VALID', scope='conv1')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool1')
                net_l = slim.conv2d(net_l, 50, [3, 3], padding='VALID', scope='conv2')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool2')
                net_l = slim.conv2d(net_l, 100, [3, 3], padding='VALID', scope='conv3')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool3')
                net_l = slim.conv2d(net_l, 200, [3, 3], padding='VALID', scope='conv4')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool4')
                net_l = slim.conv2d(net_l, 500, [3, 3], padding='VALID', scope='conv5')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool5')
                net_l = slim.conv2d(net_l, 1024, [3, 3], padding='VALID', scope='conv6')
                net_l = slim.max_pool2d(net_l, [2, 2], scope='pool6')
                net_l = slim.conv2d(net_l, 1024, [1, 1], padding='VALID', scope='pw-conv')
                net_l = slim.dropout(net_l, keep_prob, is_training=is_training, scope='dropout1')
                net_l = slim.conv2d(net_l, 500, [1, 1], padding='VALID', scope='pw-conv2')
                net_l = slim.dropout(net_l, keep_prob, is_training=is_training, scope='dropout2')
                net_l = slim.conv2d(net_l, num_classes, [1, 1], activation_fn=tf.nn.sigmoid, padding='VALID', scope='pw-conv3')

            with tf.variable_scope('branch_orig') as scope:
                # net_g = shigenet3_block(images, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_g = slim.conv2d(images, 20, [3, 3], padding='VALID', scope='conv1')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool1')
                net_g = slim.conv2d(net_g, 50, [3, 3], padding='VALID', scope='conv2')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool2')
                net_g = slim.conv2d(net_g, 100, [3, 3], padding='VALID', scope='conv3')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool3')
                net_g = slim.conv2d(net_g, 200, [3, 3], padding='VALID', scope='conv4')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool4')
                net_g = slim.conv2d(net_g, 500, [3, 3], padding='VALID', scope='conv5')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool5')
                net_g = slim.conv2d(net_g, 1024, [3, 3], padding='VALID', scope='conv6')
                net_g = slim.max_pool2d(net_g, [2, 2], scope='pool6')
                net_g = slim.conv2d(net_g, 1024, [1, 1], padding='VALID', scope='pw-conv')
                net_g = slim.dropout(net_g, keep_prob, is_training=is_training, scope='dropout1')
                net_g = slim.conv2d(net_g, 500, [1, 1], padding='VALID', scope='pw-conv2')
                net_g = slim.dropout(net_g, keep_prob, is_training=is_training, scope='dropout2')
                net_g = slim.conv2d(net_g, num_classes, [1, 1], activation_fn=tf.nn.sigmoid, padding='VALID', scope='pw-conv3')
            with tf.variable_scope('logit') as scope:
                net = tf.add(net_l, net_g) # element-wise multiply
                net = slim.flatten(net, scope='flatten')

        return net


def shigenet2_ex(images, crops, num_classes, keep_prob=0.8, is_training=False, reuse=None):
    # focus branches
    with tf.variable_scope('shigenet2', reuse=reuse) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with tf.variable_scope('branch_crop') as scope:
                # net_l = shigenet3_block(crops, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_l = shigenet2_block(crops, num_classes, keep_prob=keep_prob, is_training=is_training)

            with tf.variable_scope('branch_orig') as scope:
                # net_g = shigenet3_block(images, num_classes=num_classes, keep_prob=keep_prob, is_training=is_training)
                net_g = shigenet2_block(images, num_classes, keep_prob=keep_prob, is_training=is_training)

            with tf.variable_scope('logit') as scope:
                net = tf.multiply(net_l, net_g) # element-wise multiply
                net = slim.flatten(net, scope='flatten')

        return net

shigenet.default_input_size = vgg.vgg_16.default_image_size

def show_variables():
    print('\n'.join([v.name for v in tf.global_variables()]))

def load_batch(dataset, batch_size=5, height=shigenet.default_input_size, width=shigenet.default_input_size, is_training=False, shuffle=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=shuffle)

    image, label, bbox, fname, videoname = data_provider.get(['image', 'label', 'object/bbox', 'fname', 'videoname'])

    # preprocess and cropping
    image, cropped = lenet_preprocessing.preprocess_image(
        image,
        bbox,
        height,
        width,
        is_training)

    images, crops, labels, bboxes, fnames, videonames = tf.train.batch(
        [image, cropped, label, bbox, fname, videoname],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, crops, labels, bboxes, fnames, videonames