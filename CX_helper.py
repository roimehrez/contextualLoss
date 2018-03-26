import tensorflow as tf
from features import *
from CX import CXFlow


def CX_loss_helper(vgg_A, vgg_B, CX_config, deform_on = False):
    if CX_config.crop_quarters is True:
        vgg_A = crop_quarters(vgg_A)
        vgg_B = crop_quarters(vgg_B)

    N, fH, fW, fC = vgg_A.shape.as_list()
    if fH * fW <= CX_config.max_sampling_1d_size ** 2:
        print(' #### Skipping pooling for Diversity....')
    else:
        print(' #### pooling for Diversity %d**2 out of %dx%d' % (CX_config.max_sampling_1d_size, fH, fW))
        vgg_A, vgg_B = random_pooling([vgg_A, vgg_B], output_1d_size=CX_config.max_sampling_1d_size, patch_size=CX_config.patch_size)


    CX_loss = CXFlow.CXFlow.CX_loss(vgg_A, vgg_B, distance=CX_config.Dist, nnsigma=CX_config.nn_stretch_sigma)
    CX_loss = -tf.log(1 - CX_loss) if CX_config.log_dis else CX_loss
    CX_loss = tf.reduce_mean(CX_loss)
    return CX_loss
