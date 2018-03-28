import tensorflow as tf

from CX.CSFlow import CSFlow
from CX.enums import TensorAxis, Distance





def CX_loss(T_features, I_features, distance=Distance.L2, nnsigma=float(1.0)):
    T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)

    with tf.name_scope('CX'):
        nnw_flow = CSFlow.create(I_features, T_features, distance, nnsigma)
        # sum_normalize:
        height_width_axis = [TensorAxis.H, TensorAxis.W]
        # To:
        nnw = nnw_flow.nnw_NHWC
        k_max_NC = tf.reduce_max(nnw, axis=height_width_axis)
        CS = tf.reduce_mean(k_max_NC, axis=[1])
        CX_as_loss = 1 - CS
        return CX_as_loss




def __init__(self, sT, nnw_flow: CSFlow):
    if nnw_flow is not None:
        self.nnw_flow = nnw_flow
        self.nnw_NHWC = nnw_flow.cs_NHWC
    self.sT = sT


