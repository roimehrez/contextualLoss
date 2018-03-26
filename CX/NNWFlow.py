from logging import exception
import CX.enums as enums

import tensorflow as tf

from CX.enums import TensorAxis

class NNWFlow:
    def __init__(self, sigma = float(0.1), b = float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_NNW(self, scaled_distances, axis_for_normalization = TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.nnw_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma, name='weights_before_normalization')
        self.nnw_NHWC = NNWFlow.sum_normalize(self.nnw_weights_before_normalization, axis_for_normalization)

    def reversed_direction_NNW(self):
        nnw_flow_opposite = NNWFlow(self.sigma, self.b)
        nnw_flow_opposite.raw_distances = self.raw_distances
        work_axis = [TensorAxis.H, TensorAxis.W]
        relative_dist = nnw_flow_opposite.calc_relative_distances(axis=work_axis)
        nnw_flow_opposite.__calculate_NNW(relative_dist, work_axis)
        return nnw_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma = float(0.1), b = float(1.0)):
        nnw_flow = NNWFlow(sigma, b)
        # for debug:
        # I_features = tf.concat([I_features, I_features], axis=1)
        with tf.name_scope('NNW'):
            # assert I_features.shape[TensorAxis.C] == T_features.shape[TensorAxis.C]
            c = T_features.shape[TensorAxis.C].value
            sT = T_features.shape.as_list()
            sI = I_features.shape.as_list()

            Ivecs = tf.reshape(I_features, (sI[TensorAxis.N], -1, sI[TensorAxis.C]))
            Tvecs = tf.reshape(T_features, (sI[TensorAxis.N], -1, sT[TensorAxis.C]))
            r_Ts = tf.reduce_sum(Tvecs * Tvecs, 2)
            r_Is = tf.reduce_sum(Ivecs * Ivecs, 2)
            raw_distances_list = []
            for i in range(sT[TensorAxis.N]):
                Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
                A = Tvec @ tf.transpose(Ivec)
                nnw_flow.A = A
                # A = tf.matmul(Tvec, tf.transpose(Ivec))
                r_T = tf.reshape(r_T, [-1, 1])  # turn to column vector
                dist = r_T - 2 * A + r_I
                nnw_shape = sI[:3] + [dist.shape[0].value]
                nnw_shape[0] = 1
                dist = tf.reshape(tf.transpose(dist), nnw_shape)
                # protecting against numerical problems, dist should be positive
                dist = tf.maximum(float(0.0), dist)
                # dist = tf.sqrt(dist)
                raw_distances_list += [dist]

            nnw_flow.raw_distances = tf.convert_to_tensor([tf.squeeze(raw_dist, axis=0) for raw_dist in raw_distances_list])

            relative_dist = nnw_flow.calc_relative_distances()
            nnw_flow.__calculate_NNW(relative_dist)
            return nnw_flow

    #--
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma = float(1.0), b = float(1.0)):
        nnw_flow = NNWFlow(sigma, b)
        with tf.name_scope('NNW'):
            # prepare feature before calculating cosine distance
            T_features, I_features = nnw_flow.center_by_T(T_features, I_features)
            with tf.name_scope('TFeatures'):
                T_features = NNWFlow.l2_normalize_channelwise(T_features)
            with tf.name_scope('IFeatures'):
                I_features = NNWFlow.l2_normalize_channelwise(I_features)

                # work seperatly for each example in dim 1
                cosine_dist_l = []
                N, _, __, ___ = T_features.shape.as_list()
                for i in range(N):
                    T_features_i = tf.expand_dims(T_features[i, :, :, :], 0)
                    I_features_i = tf.expand_dims(I_features[i, :, :, :], 0)
                    patches_HWCN_i = nnw_flow.patch_decomposition(T_features_i)
                    cosine_dist_i = tf.nn.conv2d(I_features_i, patches_HWCN_i, strides=[1, 1, 1, 1],
                                                        padding='VALID', use_cudnn_on_gpu=True, name='cosine_dist')
                    cosine_dist_l.append(cosine_dist_i)

                nnw_flow.cosine_dist = tf.concat(cosine_dist_l, axis = 0)

                cosine_dist_zero_to_one = -(nnw_flow.cosine_dist - 1) / 2
                nnw_flow.raw_distances = cosine_dist_zero_to_one

                relative_dist = nnw_flow.calc_relative_distances()
                nnw_flow.__calculate_NNW(relative_dist)
                return nnw_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = tf.reduce_min(self.raw_distances, axis=axis, keep_dims=True)
        # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    def weighted_average_dist(self, axis = TensorAxis.C):
        if not hasattr(self, 'raw_distances'):
            raise exception('raw_distances property does not exists. cant calculate weighted average l2')

        multiply = self.raw_distances * self.nnw_NHWC
        return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')

    # --
    @staticmethod
    def create(I_features, T_features, distance : enums.Distance, nnsigma=float(1.0), b=float(1.0)):
        if distance.value == enums.Distance.DotProduct.value:
            nnw_flow = NNWFlow.create_using_dotP(I_features, T_features, nnsigma, b)
        elif distance.value == enums.Distance.L2.value:
            nnw_flow = NNWFlow.create_using_L2(I_features, T_features, nnsigma, b)
        else:
            raise "not supported distance " + distance.__str__()
        return nnw_flow

    @staticmethod
    def sum_normalize(nnw, axis=TensorAxis.C):
        reduce_sum = tf.reduce_sum(nnw, axis, keep_dims=True, name='sum')
        return tf.divide(nnw, reduce_sum, name='sumNormalized')

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size

        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT, self.varT = tf.nn.moments(
            T_features, axes, name='TFeatures/moments')

        # we do not divide by std since its causing the histogram
        # for the final nnw to be very thin, so the NN weights
        # are not distinctive, giving similar values for all patches.
        # stdT = tf.sqrt(varT, "stdT")
        # correct places with std zero
        # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)

        # TODO check broadcasting here
        with tf.name_scope('TFeatures/centering'):
            self.T_features_centered = T_features - self.meanT
        with tf.name_scope('IFeatures/centering'):
            self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered
    @staticmethod

    def l2_normalize_channelwise(features):
        norms = tf.norm(features, ord='euclidean', axis=TensorAxis.C, name='norm')
        # expanding the norms tensor to support broadcast division
        norms_expanded = tf.expand_dims(norms, TensorAxis.C)
        features = tf.divide(features, norms_expanded, name='normalized')
        return features

    def patch_decomposition(self, T_features):
        # patch decomposition
        # see https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
        patch_size = 1
        patches_as_depth_vectors = tf.extract_image_patches(
            images=T_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')

        self.patches_NHWC = tf.reshape(
            patches_as_depth_vectors,
            shape=[-1, patch_size, patch_size, patches_as_depth_vectors.shape[3].value],
            name='patches_PHWC')

        self.patches_HWCN = tf.transpose(
            self.patches_NHWC,
            perm=[1, 2, 3, 0],
            name='patches_HWCP')  # tf.conv2 ready format

        return self.patches_HWCN