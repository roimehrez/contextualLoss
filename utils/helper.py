# ---------------------------------------------------
#   code credits: https://github.com/CQFIO/PhotographicImageSynthesis
# ---------------------------------------------------
import numpy as np
import scipy
from config import *
import tensorflow as tf


def read_image(file_name, resize=True, fliplr=False):
    image = scipy.misc.imread(file_name)
    if resize:
        image = scipy.misc.imresize(image, size=config.TRAIN.resize, interp='bilinear', mode=None)
    if fliplr:
        image = np.fliplr(image)
    image = np.float32(image)
    return np.expand_dims(image, axis=0)


def save_image(output, file_name):
    output = np.minimum(np.maximum(output, 0.0), 255.0)
    scipy.misc.toimage(output.squeeze(axis=0), cmin=0, cmax=255).save(file_name)
    return


def write_loss_in_txt(loss_list, epoch):
    target = open(config.TRAIN.out_dir + "/%04d/score.txt" % epoch, 'w')
    target.write("%f" % np.mean(loss_list[np.where(loss_list)]))
    target.close()


def random_crop_together(im1, im2, size):
    images = tf.concat([im1, im2], axis=0)
    images_croped = tf.random_crop(images, size=size)
    im1, im2 = tf.split(images_croped, 2, axis=0)
    return im1, im2

