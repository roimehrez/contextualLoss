import os
from easydict import EasyDict as edict
import json
from CX.enums import Distance
import tensorflow as tf
import numpy as np
import re

celebA = False
single_image = False
zero_tensor = tf.constant(0.0, dtype=tf.float32)
config = edict()

#---------------------------------------------
#               update the right paths
config.base_dir = 'C:/DATA/person2person/single/'
config.single_image_B_file_name = 'images/trump_cartoon.jpg'
config.vgg_model_path = 'C:/DATA/VGG_Model/imagenet-vgg-verydeep-19.mat'
#---------------------------------------------



config.W = edict()
# weights
config.W.CX = 1.0
config.W.CX_content = 1.0

# train parameters
config.TRAIN = edict()
config.TRAIN.is_train = True #change to True of you want to train
config.TRAIN.sp = 256
config.TRAIN.aspect_ratio = 1  # 1
config.TRAIN.resize = [config.TRAIN.sp * config.TRAIN.aspect_ratio, config.TRAIN.sp]
config.TRAIN.crop_size = [config.TRAIN.sp * config.TRAIN.aspect_ratio, config.TRAIN.sp]
config.TRAIN.A_data_dir = 'train'
config.TRAIN.out_dir = "result/"
config.TRAIN.num_epochs = 10
config.TRAIN.reduce_dim = 2 #use of smaller CRN model
config.TRAIN.every_nth_frame = 1 #train using all frames

config.VAL = edict()
config.VAL.A_data_dir = 'test'
config.VAL.every_nth_frame = 1

config.TEST = edict()
config.TEST.is_test = not config.TRAIN.is_train
config.TEST.A_data_dir = config.VAL.A_data_dir
# config.TEST.every_nth_frame = 5
config.TEST.out_dir_postfix = "/test"
config.TEST.random_crop = False # if False, take the top left corner

config.CX = edict()
config.CX.crop_quarters = False
config.CX.max_sampling_1d_size = 65
# config.dis.feat_layers = {'conv1_1': 1.0,'conv2_1': 1.0, 'conv3_1': 1.0, 'conv4_1': 1.0,'conv5_1': 1.0}
config.CX.feat_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
config.CX.feat_content_layers = {'conv4_2': 1.0}  # for single image
config.CX.Dist = Distance.DotProduct
config.CX.nn_stretch_sigma = 0.5#0.1
config.CX.patch_size = 5
config.CX.patch_stride = 2


def last_two_nums(str):
    if str.endswith('vgg_input_im') or str is 'RGB':
        return 'rgb'
    all_nums = re.findall(r'\d+', str)
    return all_nums[-2] + all_nums[-1]





config.expirament_postfix = 'single_im'
if config.W.CX > 0:
    config.expirament_postfix += "_CXt" #CX_target
    config.expirament_postfix += '_'.join([last_two_nums(layer) for layer in sorted(config.CX.feat_layers.keys())])
    config.expirament_postfix += '_{}'.format(config.W.CX)
if config.W.CX_content:
    config.expirament_postfix += "_CXs" #CX_source
    config.expirament_postfix += '_'.join([last_two_nums(layer) for layer in sorted(config.CX.feat_content_layers.keys())])
    config.expirament_postfix += '_{}'.format(config.W.CX_content)


# uncomment and update for test
# config.expirament_postfix = 'm2f_D32_42_1.0(s0.5)_DC42_1.0'

config.TRAIN.out_dir += config.expirament_postfix
config.TEST.out_dir = config.TRAIN.out_dir
if not os.path.exists(config.TRAIN.out_dir):
    os.makedirs(config.TRAIN.out_dir)


