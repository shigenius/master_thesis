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


def shigenet(images, crops, num_classes, is_training=False, reuse=None):
    with tf.variable_scope('shigenet', reuse=reuse) as scope:
        arg_scope = vgg.vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('extractor', reuse=None) as scope:
                logits_l, end_points_l = vgg.vgg_16(crops, num_classes=1000, is_training=False)
                feature_l = end_points_l['shigenet/extractor/vgg_16/conv5/conv5_3']
            with tf.variable_scope('extractor', reuse=True) as scope:
                logits_g, end_points_g = vgg.vgg_16(images, num_classes=1000, is_training=False)
                print(end_points_g)
                feature_g = end_points_g['shigenet/extractor/vgg_16/conv5/conv5_3']
            # with tf.variable_scope('branch_local') as scope:
            #
            # with tf.variable_scope('branch_global') as scope:
            with tf.variable_scope('logit') as scope:
                print(feature_g)
                net = tf.concat([feature_g, feature_l], 3)
                net = slim.conv2d(net, 1024, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = slim.flatten(net, scope='flatten')
                net = slim.fully_connected(net, 500, scope='fc1')
                net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc2')
        return net


shigenet.default_input_size = vgg.vgg_16.default_image_size


def load_batch(dataset, batch_size=5, height=shigenet.default_input_size, width=shigenet.default_input_size, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=True)

    image, label, bbox, fname = data_provider.get(['image', 'label', 'object/bbox', 'fname'])

    # preprocess and cropping
    image, cropped = lenet_preprocessing.preprocess_image(
        image,
        bbox,
        height,
        width,
        is_training)

    images, crops, labels, bboxes = tf.train.batch(
        [image, cropped, label, bbox],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, crops, labels, bboxes