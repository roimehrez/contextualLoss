import tensorflow as tf

from CX.enums import TensorAxis


class DDISLoss:
    def __init__(self, T_ph, I_ph, ddis_obj, tv_lamda=0, lowres_lamda=0, max_ddis_lamda=0, weighted_l2_lamda=0):
        self.ddis_obj = ddis_obj
        self.lowres_lamda = lowres_lamda
        self.tv_lamda = tv_lamda
        self.max_ddis_lamda = max_ddis_lamda
        self.weighted_l2_lamda = weighted_l2_lamda
        self.T_ph = T_ph
        self.I_ph = I_ph
        self.ddis_loss_tensor = self.create_loss()
        self.I_ph_grad = tf.gradients(self.ddis_loss_tensor, [I_ph])[0]

    @property
    def ddis_map(self):
        return self.ddis_obj.ddis_map

    @property
    def ddis_score_tensor(self):
        return self.ddis_obj.output_score

    def create_loss(self):
        # loss = 0
        # loss = -self.ddis_obj.diversity_score
        loss = -self.ddis_score_tensor

        if self.tv_lamda > 0:
            self.tv = tf.image.total_variation(self.I_ph) / (self.I_ph.shape[1].value*self.I_ph.shape[2].value)
            # self.tv = tf.abs(tf.image.total_variation(self.I_ph) - tf.image.total_variation(self.T_ph))
            # self.tv = tf.reduce_sum(tf.abs(I_ph - T_ph), axis=[0,1,2,3])  # L1
            # self.tv = tf.constant(0, dtype=tf.float32)
            loss += self.tv_lamda * self.tv

        if self.lowres_lamda > 0:
            Ismall = DDISLoss.avg_pool(self.I_ph)
            Tsmall = DDISLoss.avg_pool(self.T_ph)

            # imsize = 56
            # Ismall = tf.image.resize_bilinear(I_ph, (imsize, imsize))
            # Tsmall = tf.image.resize_bilinear(T_ph, (imsize, imsize))
            L1_small = tf.reduce_sum(tf.abs(Ismall - Tsmall), axis=[0, 1, 2, 3])  # L1
            self.lowres_comparison_loss = L1_small
            loss += self.lowres_lamda * L1_small

        if self.weighted_l2_lamda > 0:
            l2sqr_sum = tf.reduce_sum(self.ddis_obj.nnw_flow.weighted_average_dist(), axis = [TensorAxis.H, TensorAxis.W])
            # loss += self.weighted_l2_lamda * tf.sqrt(l2sqr_sum)
            loss += self.weighted_l2_lamda * l2sqr_sum

        if self.max_ddis_lamda > 0:
            loss += self.max_ddis_lamda * tf.reduce_max(tf.maximum(tf.constant(float(0)), self.ddis_map - 1), axis=[TensorAxis.H, TensorAxis.W])

        return loss

    @staticmethod
    def avg_pool(I):
        I = tf.nn.avg_pool(tf.abs(I), [1, 20, 20, 1], strides=[1, 1, 1, 1], padding='VALID')
        # I = tf.nn.avg_pool(I, [1, 5, 5, 1], strides=[1, 2, 2, 1], padding='VALID')
        return I

    def eval(self, sess, T, I):
        if sess is None:
            sess = tf.get_default_session()
        eval_ddis_score, eval_ddis_loss, eval_I_ph_grad, eval_tv = sess.run(
            fetches=[
                self.ddis_score_tensor,
                self.ddis_loss_tensor,
                self.I_ph_grad,
                self.tv
            ],
            feed_dict={
                self.T_ph: T,
                self.I_ph: I
            }
        )
        return eval_ddis_loss, eval_I_ph_grad
