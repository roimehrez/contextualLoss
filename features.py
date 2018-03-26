import tensorflow as tf
import numpy as np
import scipy


def crop_img(img_pre, xcrop, ycrop):
    "Function to crop center of an image file"
    if isinstance(img_pre, np.ndarray):
        _, ysize, xsize, chan = img_pre.shape
    else:
        _, ysize, xsize, chan = img_pre.shape.as_list()
    xoff = (xsize - xcrop) // 2
    xoff_remainder = (xsize - xcrop) % 2
    yoff = (ysize - ycrop) // 2
    yoff_remainder = (ysize - ycrop) % 2
    img = img_pre[:, yoff:-yoff - yoff_remainder, xoff:-xoff - xoff_remainder, :]
    return img


def random_crop_together(im1, im2, size):
    size = size.copy()
    size[0] += im1.shape[0].value
    images = tf.concat([im1, im2], axis=0)
    images_croped = tf.random_crop(images, size=size)
    im1, im2 = tf.split(images_croped, 2, axis=0)
    return im1, im2


def random_sampling(tensor_NHWC, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_NHWC).shape.as_list()
    # samples = []
    # for i in range(n):
    #     sample_N_P_P_C = tf.random_crop(tensor_NHWC, [N, patch_size, patch_size, C])
    #     sample_N_PP_C = tf.reshape(sample_N_P_P_C, [N,patch_size**2, C])
    #     samples.append(sample_N_PP_C)
    #
    # return tf.concat(samples, axis=1)

    S = H * W
    tensor_NSC = tf.reshape(tensor_NHWC, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random_shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
    indices_old = tf.random_uniform([n], 0, S, tf.int32) if indices is None else indices
    res = tf.gather(tensor_NSC, indices, axis=1)
    return res, indices


def random_pooling(feats, patch_size=1, stride=1, output_1d_size=100):
    is_input_tensor = type(feats) is tf.Tensor

    if is_input_tensor:
        feats = [feats]

    # convert all inputs to tensors
    feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]

    N, H, W, C = feats[0].shape.as_list()
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]
    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)

    res = [tf.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    if is_input_tensor:
        return res[0]
    return res


def build_feature_tensor_for_diversity(vgg_layers, layerWeights):
    list = []
    for layerName, weight in layerWeights.items():
        list.append(weight * vgg_layers[layerName].outputs)
    return tf.concat(list, axis=3)


def crop_quarters(feature_tensor_target):
    N, fH, fW, fC = feature_tensor_target.shape.as_list()
    list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    list.append(tf.slice(feature_tensor_target, [0, 0, 0, 0], quarter_size))
    list.append(tf.slice(feature_tensor_target, [0, round(fH / 2), 0, 0], quarter_size))
    list.append(tf.slice(feature_tensor_target, [0, 0, round(fW / 2), 0], quarter_size))
    list.append(tf.slice(feature_tensor_target, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor_target = tf.concat(list, axis=0)
    return feature_tensor_target


def __gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def blurRGB(x, kernel_size=21):
    kernel_var = __gauss_kernel(kernel_size, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')


def extract_patch_features(I_ph, patch_size, stride, name='patches_as_depth_vectors'):
    I_features_tensor = tf.extract_image_patches(
        images=I_ph, ksizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID',
        name=name)
    return I_features_tensor
