import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.contrib.distributions as tfd
import numpy as np


class USC:

    def __init__(self, batch_size, num_markers, k, loss_coef_kernel, loss_coef_depict, trainings_data,
                 learning_rate, code_size, cluster_network, gradient_stop_on_data=False, predict_class_similarity=False):
        self.model_hparams = dict()
        self.model_hparams['batch_size'] = batch_size
        self.model_hparams['k'] = k
        self.model_hparams['loss_coef_kernel'] = loss_coef_kernel
        self.model_hparams['loss_coef_depict'] = loss_coef_depict
        self.model_hparams['learning_rate'] = learning_rate
        self.model_hparams['code_size'] = code_size
        self.data_train = trainings_data
        self.data_predict = trainings_data
        if gradient_stop_on_data:
            self.data_train = tf.stop_gradient(self.data_train)
            self.data_predict = tf.stop_gradient(self.data_predict)
        self.cluster_network = cluster_network
        self.predict_class_similarity = predict_class_similarity

        self.__define_model()




    def __create_variables(self):
        self.k_ij = tf.placeholder(tf.float32, shape=[self.model_hparams['batch_size'], self.model_hparams['batch_size']], name='k_ij')
        if self.predict_class_similarity:
            self.class_similarity = tf.placeholder(dtype=tf.float32, shape=[self.model_hparams['batch_size'], self.model_hparams['k']], name='class_similarity')

        self.w_ik_predict = tf.Variable(1, validate_shape=False, dtype=tf.float32)
        self.w_ik = tf.get_variable(dtype=tf.float32, name='prob_assignment', shape=[self.model_hparams['batch_size'], self.model_hparams['k']], validate_shape=True, initializer=xavier_initializer())
        self.alpha_k = tf.get_variable(dtype=tf.float32, name='mixture_weights', shape=[self.model_hparams['k']], validate_shape=True, initializer=xavier_initializer())
        self.mu_k = tf.get_variable(dtype=tf.float32, name='mean', shape=[self.model_hparams['k'], self.model_hparams['code_size']], validate_shape=True, initializer=xavier_initializer())
        self.sigma_k = tf.get_variable(dtype=tf.float32, name='covariance', shape=[self.model_hparams['k'], self.model_hparams['code_size']], validate_shape=True, initializer=xavier_initializer())

        self.softmax_parameter_represenation = tf.get_variable(dtype=tf.float32, name='depict_represenation', shape=[self.model_hparams['code_size'], self.model_hparams['k']])


    def __define_model(self):

        self.__create_variables()

        ##################################################
        # predictions
        preds_train, reps_train, preds, reps, reps_train_noisy = self.cluster_network(self.data_train, self.data_predict)

        self.w_ik = preds_train
        self.representation = reps_train
        self.w_ik = tf.keras.activations.softmax(self.w_ik)
        self.w_ik_new_similarity = tf.matmul(self.w_ik, self.w_ik, transpose_b=True)
        w_ik_new_assignments = tf.reduce_sum(self.w_ik, axis=0) / self.model_hparams['batch_size']
        self.cluster_assignments = tf.argmax(self.w_ik, axis=1)

        w_ik_new_predict = preds
        self.representation_predict = reps
        self.w_ik_predict = tf.keras.activations.softmax(w_ik_new_predict)

        ##################################################
        # depict auxilary fct on cluster predictions
        q_i_k_norm_factor_sum_code = tf.sqrt(tf.reduce_sum(tf.clip_by_value(self.w_ik, 1e-30, 1), axis=0))
        q_ik = tf.squeeze(tf.map_fn(lambda x: x / q_i_k_norm_factor_sum_code, self.w_ik, dtype=tf.float32))
        q_i_k_norm_factor_code_noisy = tf.reduce_sum(q_ik, axis=1)
        q_ik = q_ik / tf.stack([q_i_k_norm_factor_code_noisy] * self.model_hparams['k'], axis=1)
        q_ik = tf.clip_by_value(q_ik, 1e-30, 1)

        ##################################################
        # depict on cluster representation
        self.loss_depict_represenation = tf.constant(0, dtype=tf.float32)
        if not reps_train_noisy is None:
            p_i_k_code = tf.exp(tf.matmul(reps_train, self.softmax_parameter_represenation))
            p_i_k_norm_factor_code = tf.reduce_sum(p_i_k_code, axis=1)
            p_i_k_code = p_i_k_code / tf.stack([p_i_k_norm_factor_code]*self.model_hparams['k'], axis=1)
            p_i_k_code = tf.clip_by_value(p_i_k_code, 1e-30, 1)

            p_i_k_code_noisy = tf.exp(tf.matmul(reps_train_noisy, self.softmax_parameter_represenation))
            p_i_k_norm_factor_code = tf.reduce_sum(p_i_k_code_noisy, axis=1)
            p_i_k_code_noisy = p_i_k_code_noisy / tf.stack([p_i_k_norm_factor_code]*self.model_hparams['k'], axis=1)
            p_i_k_code_noisy = tf.clip_by_value(p_i_k_code_noisy, 1e-30, 1)

            q_i_k_norm_factor_sum_code = tf.sqrt(tf.reduce_sum(p_i_k_code_noisy, axis=0))
            q_i_k_code_noisy = tf.squeeze(tf.map_fn(lambda x : x / q_i_k_norm_factor_sum_code, p_i_k_code_noisy, dtype=tf.float32))
            q_i_k_norm_factor_code_noisy = tf.reduce_sum(q_i_k_code_noisy, axis=1)
            q_i_k_code_noisy = q_i_k_code_noisy / tf.stack([q_i_k_norm_factor_code_noisy]*self.model_hparams['k'], axis=1)
            q_i_k_code_noisy = tf.clip_by_value(q_i_k_code_noisy, 1e-30, 1)

            self.loss_depict_represenation = - 1.0/self.model_hparams['batch_size'] * tf.reduce_sum(tf.reduce_sum(q_i_k_code_noisy * tf.log(p_i_k_code), axis=1), axis=0)



        ##################################################
        # loss fctions
        similarity_cluster_prob = tf.matmul(self.k_ij, self.w_ik)
        similarity_cluster_prob = tf.keras.layers.Softmax(axis=1)(similarity_cluster_prob)
        self.loss_entropy_batch_sample = tf.reduce_mean(-1*tf.reduce_sum(similarity_cluster_prob * tf.log(similarity_cluster_prob), axis=1))

        self.loss_depict = - 1/ self.model_hparams['batch_size'] * tf.reduce_sum(tf.reduce_sum(q_ik * tf.log(tf.clip_by_value(self.w_ik, 1e-30, 1)), axis=1), axis=0)

        if self.predict_class_similarity:
            self.loss_class_similarity = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=self.class_similarity, output=self.w_ik))
        else:
            self.loss_kernel_similarity = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=self.k_ij, output=self.w_ik_new_similarity))


        if self.predict_class_similarity:
            self.loss_clustering = self.model_hparams['loss_coef_kernel'] * self.loss_class_similarity + \
                                   self.model_hparams['loss_coef_depict'] * self.loss_depict + \
                                   self.model_hparams['loss_coef_depict'] * self.loss_depict_represenation
        else:
            self.loss_clustering = self.model_hparams['loss_coef_kernel'] * self.loss_kernel_similarity + \
                                   self.model_hparams['loss_coef_depict'] * self.loss_depict + \
                                   self.model_hparams['loss_coef_depict'] * self.loss_depict_represenation
        self.optimize_clustering = tf.train.AdamOptimizer(self.model_hparams['learning_rate'], beta1=0.9, beta2=0.99).minimize(self.loss_clustering)

        ##################################################
        # M-step
        self.N_k = tf.clip_by_value(tf.reduce_sum(self.w_ik, axis=0), 1, self.model_hparams['batch_size'])
        alpha_k_new = self.N_k / self.model_hparams['batch_size']
        self.update_alpha_k = tf.assign(self.alpha_k, alpha_k_new)

        mu_k_new_unnormalized = tf.matmul(self.w_ik, self.data_train, transpose_a=True)
        mu_k_new_norm_factor = tf.stack([self.N_k] * self.model_hparams['code_size'], axis=1)
        mu_k_new = mu_k_new_unnormalized / mu_k_new_norm_factor
        self.updated_mu_k = tf.assign(self.mu_k, mu_k_new)

        self.log_likelihood = self.__gmm_log_likelihood(self.data_train)



    def __pairwise_euclidean_distance(self, A, B):
        """
        Computes pairwise distances between each elements of A and each elements of B.
        Args:
          A,    [m,d] matrix
          B,    [n,d] matrix
        Returns:
          D,    [m,n] matrix of pairwise distances
        """
        with tf.variable_scope('pairwise_dist'):
            # squared norms of each row in A and B
            na = tf.reduce_sum(tf.square(A), 1)
            nb = tf.reduce_sum(tf.square(B), 1)

            # na as a row and nb as a co"lumn vectors
            na = tf.reshape(na, [-1, 1])
            nb = tf.reshape(nb, [1, -1])

            # return pairwise euclidead difference matrix
            D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
        return D

    def __gmm_log_likelihood(self, data):
        # evaluate based on log likelihood
        mixture_probabilities = tf.transpose(
            tf.map_fn(lambda x: tfd.MultivariateNormalDiag(loc=x[0], scale_diag=x[1]).prob(data) * x[2],
                      (self.mu_k, self.sigma_k, self.alpha_k), dtype=tf.float32))
        log_likelihood = tf.reduce_sum(tf.log(tf.reduce_sum(mixture_probabilities, axis=1)), axis=0)
        return log_likelihood
