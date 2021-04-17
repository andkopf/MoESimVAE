import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from src.utils import initializers, activation_fcts


class Encoder(object):


    def __init__(self, code_size, encoder_internal_size, activation_fct, dropout_rate, init_method='xavier', batch_normalization=True, depth_encoder=1):
        self.code_size = code_size
        self.encoder_internal_size = encoder_internal_size
        self.batch_normalization = batch_normalization
        self.init = initializers[init_method]
        self.activation_fct = activation_fcts[activation_fct]
        self.depth_encoder = depth_encoder
        self.dropout_rate = dropout_rate

    def build_encoder(self, x):
        x = tf.layers.flatten(x)
        noisy_x = tf.layers.dropout(x, rate=self.dropout_rate)

        for i in range(self.depth_encoder):
            my_dense = tf.layers.Dense(self.encoder_internal_size, kernel_initializer=self.init(), name="denes_" + str(i))
            x = my_dense(x)
            noisy_x = my_dense(noisy_x)
            if self.batch_normalization:
                x = tf.layers.batch_normalization(x)
                noisy_x = tf.layers.batch_normalization(noisy_x)
            x = self.activation_fct(x)
            noisy_x = self.activation_fct(noisy_x)

        self.loc = tf.layers.dense(x, self.code_size, kernel_initializer=self.init())
        self.loc_noisy = tf.layers.dense(noisy_x, self.code_size, kernel_initializer=self.init())
        self.scale = tf.ones(tf.shape(self.loc), dtype=tf.float32)
        return tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.scale).sample(), \
               tfd.MultivariateNormalDiag(loc=self.loc_noisy, scale_diag=self.scale).sample(), self.loc, self.loc_noisy, self.scale

