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


def shigenet(images, crops, num_classes):
    with tf.variable_scope('shigenet') as scope:
        arg_scope = vgg.vgg_arg_scope()
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('branch_local') as scope:
                net_l, _ = vgg.vgg_16(crops, num_classes=num_classes)
            with tf.variable_scope('branch_global') as scope:
                net_g, _ = vgg.vgg_16(images, num_classes=num_classes)

            net = tf.concat([net_l, net_g], 1)
            net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc')
            print(net)

        return net

shigenet.default_input_size = vgg.vgg_16.default_image_size


def load_batch(dataset, batch_size=5, height=224, width=224, is_training=False):
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