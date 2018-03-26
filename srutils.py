import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

from tensorlayer.prepro import *

def pp(x):
    print(x) #todo del func

def load_and_assign_latest_checkpoint(sess, checkpoint_dir, prefix, network):
    extract_epoch_func = lambda name: int(re.search(r'\d+', name + '__default_epoch_is_0').group())
    all_epochs = sorted(tl.files.load_file_list(path=checkpoint_dir, regx='{}.*.npz'.format(prefix), printable=False), key=extract_epoch_func)
    if len(all_epochs) == 0:
        return False

    last_epoch = all_epochs[-1]
    last_stop_checkpoint = extract_epoch_func(last_epoch)
    loaded = False if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + last_epoch, network=network) is False else True
    return loaded

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx: idx + n_threads]
        b_imgs = threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')


def crop_and_rescale_old(x, is_random=True):
    return crop_and_rescale2(x, is_random, crop_sz=384)

def crop_and_rescale(x, is_random=True):
    # next code randomly rescale an image
    smallest_resize_ratio = np.ceil(384 / np.asarray(x.shape)[:2].min())
    range = [np.maximum(0.75, smallest_resize_ratio), 1]
    random_scale = range[0] + (range[1] - range[0]) * np.random.rand()
    h, w, c = x.shape
    new_size = [int(round(random_scale * h)), int(round(random_scale * w))]
    x = imresize(x, size=new_size, interp='bicubic', mode=None)
    return crop_and_rescale2(x, is_random, crop_sz=384)

def crop_and_rescale2(x, is_random=True, crop_sz=384):
    x = crop(x, wrg=crop_sz, hrg=crop_sz, is_random=is_random)
    x = rescale_rgb255_around_zero(x)
    return x

def downsample_fn(x,h=96,w=96):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[h, w], interp='bicubic', mode=None)
    # fixing auto rescale done by imresize:
    x = rescale_rgb255_around_zero(x)
    return x

def downsample_fn_512_256(x,h=512,w=256):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[h, w], interp='bicubic', mode=None)
    # fixing auto rescale done by imresize:
    x = rescale_rgb255_around_zero(x)
    return x


def rescale_rgb255_around_zero(x):
    return (x / 127.5) - 1

def rescale_around_zero_to_rgb255(x):
    return rescale_around_zero_to_01(x) * 255.

def rescale_around_zero_to_01(x):
    return (x+1) / 2.

def imsave_with_retries(img, size, filepath):
    for kk in range(5):
        try:
            tl.vis.save_images(img, size, filepath)
            break
        except:
            # print('failed')
            time.sleep(1)


def elements_in_layer(feature_tensor):
    n, h, w, c = feature_tensor.shape.as_list()
    return n*h*w*c



