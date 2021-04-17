import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from src.moe_utils import SparseDispatcher, flatten_all_but_last, Parallelism
import six
import numpy as np
from sklearn.metrics import f1_score
import argparse

xav_init = tf.contrib.layers.xavier_initializer
normal_init = tf.truncated_normal_initializer
zero_init = tf.zeros_initializer

# Different methods for initializing the data
initializers = dict()
initializers['xavier'] = xav_init
initializers['normal'] = normal_init
initializers['zeros'] = zero_init

# Different activation functions
activation_fcts = dict()
activation_fcts['relu'] = tf.keras.activations.relu
activation_fcts['elu'] = tf.keras.activations.elu
activation_fcts['sigmoid'] = tf.keras.activations.sigmoid
activation_fcts['hard_sigmoid'] = tf.keras.activations.hard_sigmoid
activation_fcts['selu'] = tf.keras.activations.selu
activation_fcts['softmax'] = tf.keras.activations.softmax
activation_fcts['softplus'] = tf.keras.activations.softplus
activation_fcts['softsign'] = tf.keras.activations.softsign
activation_fcts['tanh'] = tf.keras.activations.tanh
activation_fcts['LeakyRelu'] = tf.nn.leaky_relu




def save_loss_plot(out_dir, loss, log=False, tag=''):
    """
    Saves loss plot to output directory
    :param out_dir: str, output directory
    :param disc_loss: list, discriminator losses
    :param gen_loss: list, generator losses
    :return: no returns
    """
    if log:
        filename = os.path.join(out_dir, 'loss_log_plot'+tag+'.pdf')
    else:
        filename = os.path.join(out_dir, 'loss_plot'+tag+'.pdf')
    plt.figure()
    if log:
        plt.plot(range(len(loss)), np.log(loss), 'r')
    else:
        plt.plot(range(len(loss)), loss, 'r')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def write_readme_file(params, dir_output, filename='ReadMe.txt'):
    fout = os.path.join(dir_output, filename)
    fo = open(fout, 'w')
    for k, v in params.items():
        fo.write(str(k) + ' = ' + str(v) + '\n\n')
    fo.close()
    

def generate_subset(inputs,
                    num_cells_per_input,
                    batch_size,
                    weights=None,
                    return_indices=False):
    """
    Returns a random subset from input data of shape (batch_size, num_cells_per_input, num_markers)
    :param inputs: numpy array, the input ndarray to sample from
    :param num_cells_per_input: int, number of cells per multi-cell input
    :param batch_size: int, batch size of the subset
    :param weights: list of float, whether there is a preference for some cells
    :param return_indices: bool, whether to return subset indices or not
    :return:
    """

    num_cells_total = inputs.shape[0]

    if weights is not None:
        indices = np.random.choice(
            num_cells_total,
            size=batch_size * num_cells_per_input,
            replace=True,
            p=weights)

    else:
        indices = np.random.choice(
            num_cells_total,
            size=batch_size * num_cells_per_input,
            replace=True)

    subset = inputs[indices, ]
    subset = np.reshape(subset, newshape=(batch_size, num_cells_per_input, -1))

    if return_indices:
        return subset, indices

    else:
        return subset


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def compute_f_measure(y_true, y_pred):
    """
    Compute f-measure of subpopulation prediction results.
    :param y_true:
    :param y_pred:
    :return: double, f-measure
    """

    y_true_unique = np.unique(y_true)
    y_pred_unique = np.unique(y_pred)

    N = len(y_true)
    f_measure_i = list()

    for i, y_i in enumerate(y_true_unique):
        f_measure_j = list()
        temp_ind_y = np.where(np.asarray(y_true) == y_i)[0]

        binary_y_i = np.zeros((N, ))
        binary_y_i[temp_ind_y] = 1

        n_c_i = len(temp_ind_y)
        for j, y_j in enumerate(y_pred_unique):
            temp_ind_y_j = np.where(np.asarray(y_pred) == y_j)[0]

            binary_y_j = np.zeros((N,))
            binary_y_j[temp_ind_y_j] = 1

            f_measure_j.append(f1_score(binary_y_i, binary_y_j))

        ind_max = np.argmax(np.asarray(f_measure_j))
        f_measure_i.append(n_c_i/N*f_measure_j[ind_max])

    return(np.sum(f_measure_i))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def ffn_expert_fn(output_size,
                  init,
                  name,
                  activation_fct,
                  depth=1,
                  decoder_internal_size=200,
                  batch_normalization=True):

    def my_fn(x):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            for i in range(depth):
                x = tf.layers.dense(x, decoder_internal_size, kernel_initializer=init())
                if batch_normalization:
                    x = tf.layers.batch_normalization(x)
                x = activation_fct(x)

            output = tf.layers.dense(inputs=x,
                                     units=output_size,
                                     activation=None,
                                     name='gen_output',
                                     kernel_initializer = init())
            return output

    return my_fn


def local_moe(x,
              gates,
              list_moe_func,
              name=None,
              additional_dispatch_params=None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        dispatcher = SparseDispatcher(len(list_moe_func), tf.ones((1, len(list_moe_func))))

        expert_kwargs = {}
        expert_kwargs["x"] = dispatcher.dispatch(flatten_all_but_last(x))
        for k, v in six.iteritems(additional_dispatch_params or {}):
            v = flatten_all_but_last(v)
            expert_kwargs[k] = dispatcher.dispatch(v)

        ep = Parallelism(['existing_device']*len(list_moe_func), reuse=tf.AUTO_REUSE)
        expert_outputs = ep(list_moe_func, **expert_kwargs)

        expert_outputs = tf.stack(expert_outputs)
        expert_outputs = tf.squeeze(expert_outputs)

        expert_outputs_combined = tf.reduce_sum(tf.matmul(gates, expert_outputs), axis=0)


    return expert_outputs, expert_outputs_combined
