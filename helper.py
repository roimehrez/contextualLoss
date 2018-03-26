import os, numpy as np
from os.path import dirname, exists, join, splitext
import json, scipy
from config import *
import math
import tensorflow as tf


class Dataset(object):
    def __init__(self, dataset_name):
        self.work_dir = dirname(os.path.realpath('__file__'))
        info_path = join(self.work_dir, 'datasets', dataset_name + '.json')
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.palette = np.array(info['palette'], dtype=np.uint8)


def get_semantic_map(path):
    dataset = Dataset('cityscapes')
    semantic = scipy.misc.imread(path)
    tmp = np.zeros((semantic.shape[0], semantic.shape[1], dataset.palette.shape[0]), dtype=np.float32)
    for k in range(dataset.palette.shape[0]):
        tmp[:, :, k] = np.float32((semantic[:, :, 0] == dataset.palette[k, 0]) & (semantic[:, :, 1] == dataset.palette[k, 1]) & (semantic[:, :, 2] == dataset.palette[k, 2]))
    return tmp.reshape((1,) + tmp.shape)


def print_semantic_map(semantic, path):
    dataset = Dataset('cityscapes')
    semantic = semantic.transpose([1, 2, 3, 0])
    prediction = np.argmax(semantic, axis=2)
    color_image = dataset.palette[prediction.ravel()].reshape((prediction.shape[0], prediction.shape[1], 3))
    row, col, dump = np.where(np.sum(semantic, axis=2) == 0)
    color_image[row, col, :] = 0
    scipy.misc.imsave(path, color_image)


def read_image(file_name, resize=True, fliplr=False):
    image = np.float32(scipy.misc.imread(file_name))
    if resize:
        image = scipy.misc.imresize(image, size=config.TRAIN.resize, interp='bilinear', mode=None)
    if fliplr:
        image = np.fliplr(image)
    return np.expand_dims(image, axis=0)


def read_image_resize_width(file_name):
    image = np.float32(scipy.misc.imread(file_name))
    new_shape = (int(math.floor(float(image.shape[0]) / image.shape[1] * config.TRAIN.width)), config.TRAIN.width)
    image = scipy.misc.imresize(image, size=new_shape, interp='bilinear', mode=None)
    if image.ndim == 2:
        image = np.dstack([image, image, image])
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


def crop_to_bounding_box_together(im1, im2, offset_height=0, offset_width=0, target_height=config.TRAIN.sp, target_width=config.TRAIN.sp):
    images = tf.concat([im1, im2], axis=0)
    images_croped = tf.image.crop_to_bounding_box(images, offset_height=offset_height, offset_width=offset_width, target_height=target_height, target_width=target_width)
    im1, im2 = tf.split(images_croped, 2, axis=0)
    return im1, im2


def create_mask():
    ones_square = np.ones([config.inpaint.ones_square_size, config.inpaint.ones_square_size]).astype(np.float32)
    padd_size = (config.TRAIN.sp - config.inpaint.ones_square_size) // 2
    paddings = [[padd_size, padd_size], [padd_size, padd_size]]
    mask = tf.pad(ones_square, paddings=paddings)
    mask = tf.stack([mask, mask, mask], axis=2)
    mask = tf.expand_dims(mask, axis=0)
    # mask_bool = tf.cast(mask, tf.bool)
    return mask