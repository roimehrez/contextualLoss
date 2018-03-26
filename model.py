import tensorflow.contrib.slim as slim
import numpy as np
from vgg_model import *
from config import *


# this function have been modify such that the images are portrait and not landscape
def recursive_generator(input_image, width):
    ar = config.TRAIN.aspect_ratio
    if width >= 128:
        dim = 512 // config.TRAIN.reduce_dim
    else:
        dim = 1024 // config.TRAIN.reduce_dim

    if width == 4:
        input = input_image
    else:
        downsampled_width = width // 2
        downsampled_input = tf.image.resize_area(input_image, (downsampled_width, downsampled_width // ar), align_corners=False)
        recursive_call = recursive_generator(downsampled_input, downsampled_width)
        predicted_on_downsampled = tf.image.resize_bilinear(recursive_call, (width, width // ar), align_corners=True)
        input = tf.concat([predicted_on_downsampled, input_image], 3)

    net = slim.conv2d(input, dim, [3, 3], rate=1, normalizer_fn=slim.layer_norm, activation_fn=lrelu, scope='g_' + str(width) + '_conv1')
    net = slim.conv2d(net, dim, [3, 3], rate=1, normalizer_fn=slim.layer_norm, activation_fn=lrelu, scope='g_' + str(width) + '_conv2')

    if width == config.TRAIN.sp*config.TRAIN.aspect_ratio:
        net = slim.conv2d(net, 3, [1, 1], rate=1, activation_fn=None, scope='g_' + str(width) + '_conv100')
        net = (net + 1.0) / 2.0 * 255.0
    return net


def compute_error(real, fake):
    return tf.reduce_mean(tf.abs(fake - real))


def compute_gram(feats):
    N, H, W, C = map(lambda i: i.value, feats.get_shape())
    size = H * W * C
    feats = tf.reshape(feats, (N, H * W, C))
    feats_T = tf.transpose(feats, perm=[0, 2, 1])
    return tf.matmul(feats_T, feats) / size
