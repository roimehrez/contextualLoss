from CX import CSFlow
import tensorflow as tf


def random_sampling(tensor_NHWC, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_NHWC).shape.as_list()
    S = H * W
    tensor_NSC = tf.reshape(tensor_NHWC, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random_shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
    indices_old = tf.random_uniform([n], 0, S, tf.int32) if indices is None else indices
    res = tf.gather(tensor_NSC, indices, axis=1)
    return res, indices


def random_pooling(feats, output_1d_size=100):
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


def crop_quarters(feature_tensor):
    N, fH, fW, fC = feature_tensor.shape.as_list()
    quarters_list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    quarters_list.append(tf.slice(feature_tensor, [0, 0, 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, 0, round(fW / 2), 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor = tf.concat(quarters_list, axis=0)
    return feature_tensor


def CX_loss_helper(vgg_A, vgg_B, CX_config):
    if CX_config.crop_quarters is True:
        vgg_A = crop_quarters(vgg_A)
        vgg_B = crop_quarters(vgg_B)

    N, fH, fW, fC = vgg_A.shape.as_list()
    if fH * fW <= CX_config.max_sampling_1d_size ** 2:
        print(' #### Skipping pooling for CX....')
    else:
        print(' #### pooling for CX %d**2 out of %dx%d' % (CX_config.max_sampling_1d_size, fH, fW))
        vgg_A, vgg_B = random_pooling([vgg_A, vgg_B], output_1d_size=CX_config.max_sampling_1d_size)

    CX_loss = CSFlow.CX_loss(vgg_A, vgg_B, distance=CX_config.Dist, nnsigma=CX_config.nn_stretch_sigma)
    return CX_loss
