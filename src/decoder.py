import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from src.utils import initializers, activation_fcts, local_moe, ffn_expert_fn



class Decoder(object):

    """
    Creates an object of CellGanGen, which is the generator class for CellGan.

    Args:
        - num_experts: int
            Number of experts used in the CellGan_generator

        - num_markers: int
            Number of markers used in the experiment

        - num_filters: int
            Number of filters to be used in convolutional layer

        - noisy_gating: bool
            Whether to use the noise component in gating networks

        - noise_epsilon: float
            noise threshold

        - num_top: int
            Number of top experts to use for each example

        - init_method: str, default 'xavier'
            Method of initializing the weights

    """

    def __init__(self, num_experts, num_markers, code_size, init_method='xavier', batch_normalization=True, decoder_depth=1, activation_fct='elu', decoder_internal_size=20):

        self.num_experts = num_experts
        self.num_markers = num_markers
        self.code_size = code_size


        self.batch_normalization = batch_normalization
        self.decoder_depth = decoder_depth
        self.activation_fct = activation_fcts[activation_fct]
        self.decoder_internal_size = decoder_internal_size

        self.init = initializers[init_method]


    def build_decoder(self, logits_activated, code, gates):

        # define moe function
        with tf.variable_scope('fn_gen_outputs'):
            self.moe_func = list()
            for i in range(self.num_experts):
                my_fun = ffn_expert_fn(output_size=self.num_markers, init=self.init, name='expert_' + str(i),
                                       activation_fct=self.activation_fct, depth=self.decoder_depth,
                                       decoder_internal_size=self.decoder_internal_size, batch_normalization=self.batch_normalization)
                self.moe_func.append(my_fun)

        gen_outputs = tf.map_fn(lambda x: local_moe(x[0], tf.reshape(x[1], (1, self.num_experts)), self.moe_func, name='gen_outputs'), (code, gates), dtype=(tf.float32, tf.float32))

        return gen_outputs[0], gen_outputs[1]






