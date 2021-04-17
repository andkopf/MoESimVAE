import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import sys
from datetime import datetime as dt
import pandas as pd
from shutil import copyfile

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import multivariate_normal
from collections import OrderedDict
import umap

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, ROOT_DIR)

from src.encoder import Encoder
from src.decoder import Decoder
from src.utils import initializers, sigmoid, compute_f_measure, str2bool, generate_subset, write_readme_file, activation_fcts, save_loss_plot
from src.unsupervised_similarity_clustering import USC

def main():

    parser = argparse.ArgumentParser()

    # IO parameters
    parser.add_argument('--dir_output', help='Directory where output will be generated.')
    parser.add_argument('--plotting', type=str2bool, default=True, help='Whether to always plot the updated model.')

    ## model hyperparameters
    parser.add_argument('--batch_normalization', type=str2bool, default=True, help='Boolean whether to include batch normalization,')
    parser.add_argument('--encoder_depth', type=int, default=1, help='Depth of encoder.')
    parser.add_argument('--encoder_internal_size', type=int, default=100, help='Internal size of encoder.')
    parser.add_argument('--decoder_depth', type=int, default=1, help='Depth of decoder.')
    parser.add_argument('--decoder_internal_size', type=int, default=100, help='Internal size of decoder.')
    parser.add_argument('--depth_cluster_network', type=int, default=1, help='Depth of clustering network.')
    parser.add_argument('--internal_size_cluster_network', type=int, default=100, help='Internal size of clustering network.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout_rate', type=float, default=.5, help='Dropout rate.')
    parser.add_argument('--code_size', type=int, default=20, help='Code size.')
    parser.add_argument('--num_experts', default=7, type=int, help='Number of experts in model.')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of cells per multi-cell input.')
    parser.add_argument('--batch_size_test', type=int, default=512, help='Number of samples to test model.')
    parser.add_argument('--num_iterations', type=int, default=20000, help='Number of trainings iterations.')
    parser.add_argument('--activation_fct', default='elu', type=str, choices=['relu', 'elu', 'sigmoid', 'hard_sigmoid', 'selu', 'softmax', 'softplus', 'softsign', 'tanh', 'LeakyRelu'], help='Choice of activation functions to use in decoder.')

    ## loss coefficients
    parser.add_argument('--loss_coef_reconst_data', type=float, default=1, help='Loss coefficient of reconstruction loss.')
    parser.add_argument('--loss_coef_kl_div_code_standard_gaussian', type=float, default=1, help='Locc coefficient of Kullback Leibler divergence for GMM in latent space.')
    parser.add_argument('--loss_coef_clustering', type=float, default=1, help='Loss coefficient of clustering loss.')
    parser.add_argument('--loss_coef_cluster_kernel', type=float, default=1, help='Loss coefficient of kernel similarity.')
    parser.add_argument('--loss_coef_cluster_depict', type=float, default=1, help='Loss coefficent of DEPICT loss in clustering loss.')

    ## preprocessing
    parser.add_argument('--quantile_prob', type=float, default=.2, help='Percentile for data filtering.')

    args = parser.parse_args()

    model_hparams = {
        'path_data' : os.path.join(ROOT_DIR, 'data'),
        'plotting' : args.plotting,
        'batch_normalization' : args.batch_normalization,
        'encoder_depth' : args.encoder_depth,
        'encoder_internal_size' : args.encoder_internal_size,
        'decoder_depth' : args.decoder_depth,
        'decoder_internal_size' : args.decoder_internal_size,
        'depth_cluster_network' : args.depth_cluster_network,
        'internal_size_cluster_network' : args.internal_size_cluster_network,
        'learning_rate' : args.learning_rate,
        'dropout_rate' : args.dropout_rate,
        'code_size' : args.code_size,
        'num_experts' : args.num_experts,
        'batch_size': args.batch_size,
        'batch_size_test' : args.batch_size_test,
        'num_iterations' : args.num_iterations,
        'activation_fct' : args.activation_fct,
        'loss_coef_reconst_data' : args.loss_coef_reconst_data,
        'loss_coef_kl_div_code_standard_gaussian' : args.loss_coef_kl_div_code_standard_gaussian,
        'loss_coef_clustering' : args.loss_coef_clustering,
        'loss_coef_cluster_kernel' : args.loss_coef_cluster_kernel,
        'loss_coef_cluster_depict' : args.loss_coef_cluster_depict,
        'quantile_prob' : args.quantile_prob
    }

    # Setup the output directory
    experiment_name = dt.now().strftime('%d-%m_%H-%M-%S-%f')
    random_id = np.random.choice(list(range(99999)), 1)
    experiment_name += '_' + str(format(random_id[0], '05d'))
    dir_output = os.path.join(args.dir_output, experiment_name)

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # copy executeable in output dir
    copyfile(sys.argv[0], os.path.join(dir_output, os.path.basename(sys.argv[0])))


    # load data
    data = pd.read_csv(os.path.join(model_hparams['path_data'], 'data_merged_tpm.csv.bz2'), index_col=0, compression='bz2')
    labels = pd.read_csv(os.path.join(model_hparams['path_data'], 'labels_merged.csv'), header=None).values.squeeze()
    ind_train = pd.read_csv(os.path.join(model_hparams['path_data'], 'indices_train.csv'), header=None).values.squeeze()
    ind_test = pd.read_csv(os.path.join(model_hparams['path_data'], 'indices_test.csv'), header=None).values.squeeze()
    prior = pd.read_csv(os.path.join(model_hparams['path_data'],'labels_prior_genes.csv'), dtype=np.float32)

    training_data = data.values[ind_train]
    training_labels = labels[ind_train].squeeze()
    test_data = data.values[ind_test]
    test_labels = labels[ind_test].squeeze()
    training_prior = prior.values[ind_train]

    # filter data for points without prior knowledge
    training_prior_sum = np.sum(training_prior, axis=1)
    ind_train_prior = np.where(training_prior_sum == 1)[0]
    training_data = training_data[ind_train_prior]
    training_labels = training_labels[ind_train_prior]
    training_prior = training_prior[ind_train_prior]

    training_labels_unique = np.unique(training_labels)
    test_labels_unique = np.unique(test_labels)


    # define color mapping and marker mapping
    cmap = matplotlib.cm.get_cmap('viridis')
    colors_labels = cmap(np.linspace(0, 1, len(training_labels_unique)))

    dict_colors = dict()
    dict_marker = dict()
    for i, label in enumerate(training_labels_unique):
        dict_marker[label] = '$' + str(label) + '$'
        dict_colors[label] = colors_labels[i].reshape((1, 4))


    cmap = matplotlib.cm.get_cmap('viridis')
    colors_experts = cmap(np.linspace(0, 1, model_hparams['num_experts']))


    dict_marker_experts = dict()
    dict_colors_experts = dict()
    for exp_id in range(model_hparams['num_experts']):
        dict_marker_experts[exp_id] = '$' + str(exp_id) + '$'
        dict_colors_experts[exp_id] = colors_experts[exp_id].reshape((1, 4))

    # scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(training_data)
    training_data_scaled = scaler.transform(training_data)
    test_data_scaled = scaler.transform(test_data)

    # filter vor genes with low variance
    sum_gene_count_total = np.sum(training_data, axis=0)
    var_gene = np.var(training_data, axis=0)

    ind_filter_count = np.where(sum_gene_count_total > np.quantile(sum_gene_count_total, model_hparams['quantile_prob']))[0]
    ind_filter_var = np.where(var_gene > np.quantile(var_gene, model_hparams['quantile_prob']))[0]
    ind_gene_filter = np.unique([ind_filter_count, ind_filter_var])

    training_data_scaled = training_data_scaled[:, ind_gene_filter]
    test_data_scaled = test_data_scaled[:, ind_gene_filter]

    model_hparams['num_markers'] = training_data_scaled.shape[1]

    # load pca
    pca = PCA()
    pca = pca.fit(training_data_scaled)
    pca_transform = pca.transform(training_data_scaled)

    # load umap
    um = umap.UMAP()
    um = um.fit(training_data_scaled)
    um_transform = um.transform(training_data_scaled)

    plt.figure()
    for i, sub in enumerate(training_labels_unique):
        temp_ind = np.where(training_labels == sub)[0]
        plt.scatter(um_transform[temp_ind, 0], um_transform[temp_ind, 1], s=30, marker=dict_marker[sub], c=dict_colors[sub], label=sub)
    plt.xlabel('UM1')
    plt.ylabel('UM2')
    plt.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, 'umap_data.pdf'))
    plt.close()

    plt.figure()
    for i, sub in enumerate(training_labels_unique):
        temp_ind = np.where(training_labels == sub)[0]
        plt.scatter(pca_transform[temp_ind, 0], pca_transform[temp_ind, 1], s=30, marker=dict_marker[sub], c=dict_colors[sub], label=sub)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best', fontsize=14)     
    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, 'pca_data.pdf'))
    plt.close()


    ################################
    # define model
    data = tf.placeholder(tf.float32, [None, model_hparams['num_markers']])

    # define encoder 
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        encoder = Encoder(code_size=model_hparams['code_size'], encoder_internal_size=model_hparams['encoder_internal_size'], activation_fct=model_hparams['activation_fct'], init_method='xavier', batch_normalization=model_hparams['batch_normalization'], depth_encoder=model_hparams['encoder_depth'], dropout_rate=model_hparams['dropout_rate'])
        make_encoder = tf.make_template('encoder', encoder.build_encoder)
        code, code_noisy, code_loc, code_loc_noisy, code_scale = make_encoder(data)

    # define clustering network
    with tf.variable_scope('clustering', reuse=tf.AUTO_REUSE):
            # clustering network
            def network(x, y):
                for i in range(model_hparams['depth_cluster_network']):
                    dense = tf.keras.layers.Dense(model_hparams['internal_size_cluster_network'], activation=activation_fcts[model_hparams['activation_fct']], kernel_initializer=initializers['xavier']())
                    x = dense(x)
                    y = dense(y)
                    dropout = tf.keras.layers.Dropout(model_hparams['dropout_rate'])
                    x = dropout(x)
                    y = dropout(y)
                dense = tf.keras.layers.Dense(model_hparams['num_experts'], kernel_initializer=initializers['xavier']())

                output_x = dense(x)
                output_y = dense(y)
                return output_x, x, output_y, y, None

            usc = USC(batch_size=model_hparams['batch_size'], num_markers=model_hparams['num_markers'], k=model_hparams['num_experts'], loss_coef_kernel=model_hparams['loss_coef_cluster_kernel'], loss_coef_depict=model_hparams['loss_coef_cluster_depict'], trainings_data=code, learning_rate=model_hparams['learning_rate'], code_size=model_hparams['code_size'], cluster_network=network, predict_class_similarity=True)

            centroids = usc.mu_k
            assignments = usc.cluster_assignments
            gates_assignments = tf.one_hot(assignments, model_hparams['num_experts'], axis=1)
            loss_clustering_usc = usc.loss_clustering

    ################################
    # decoder
    with tf.variable_scope('decoder'):
        decoder = Decoder(num_experts=model_hparams['num_experts'], num_markers=model_hparams['num_markers'], code_size=model_hparams['code_size'], batch_normalization=model_hparams['batch_normalization'], decoder_depth=model_hparams['decoder_depth'], activation_fct=model_hparams['activation_fct'], decoder_internal_size=model_hparams['decoder_internal_size'])
        make_decoder = tf.make_template('decoder', decoder.build_decoder)
        gen_output, gen_output_gated = make_decoder(code, code, gates_assignments)

    ################################
    # KL loss for standard gaussian with cluster mean
    centroids_expanded = tf.expand_dims(centroids, axis=0)
    code_expanded = tf.expand_dims(code, axis=1)
    dist_samples = code_expanded - centroids_expanded
    dist_samples_squared = tf.square(dist_samples)
    gates_assignments_expanded = tf.expand_dims(gates_assignments, axis=-1)
    dist_samples_mask = dist_samples_squared * gates_assignments_expanded

    std_unnormalized = tf.reduce_sum(dist_samples_mask, axis=0)
    normalizing_factors = tf.clip_by_value(tf.reduce_sum(gates_assignments_expanded, axis=0), 1, 1e20)
    std_normalized = std_unnormalized / normalizing_factors
    std_normalized = tf.where(tf.equal(std_normalized, 0.), tf.ones_like(std_normalized), std_normalized)

    kl_divergence_code_standard_gaussian = 0.5 * (tf.reduce_sum(std_normalized, axis=1) - model_hparams['code_size'] * tf.ones(model_hparams['num_experts']) - tf.reduce_sum(tf.log(std_normalized), axis=1))
    loss_kl_divergence_code_standard_gaussian = tf.reduce_mean(kl_divergence_code_standard_gaussian)

    ################################
    # reconstruction loss for data
    loss_reconst_data = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_output_gated, labels=data), axis=1))

    ################################
    # model loss function
    loss_model = model_hparams['loss_coef_reconst_data'] * loss_reconst_data + \
                 model_hparams['loss_coef_clustering'] * loss_clustering_usc + \
                 model_hparams['loss_coef_kl_div_code_standard_gaussian'] * loss_kl_divergence_code_standard_gaussian

    ################################
    # variable scopes
    variables_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    variables_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    variables_usc_clustering = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='clustering')
    variables_depict = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='depict')

    ################################
    # define optimzers
    optimize_model = tf.train.AdamOptimizer(model_hparams['learning_rate'], beta1=0.9, beta2=0.99).minimize(loss_model, var_list=[variables_encoder, variables_decoder, variables_depict, variables_usc_clustering])

    # write down current readme file
    write_readme_file(model_hparams, dir_output)

    ################################
    # start training
    list_loss_model = list()
    list_loss_reconst_data = list()
    list_loss_kl_div_standard_gaussian = list()
    list_loss_clustering_usc = list()
    readme_current_training = dict()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for iteration in range(model_hparams['num_iterations']):
            real_batch, indices_batch = generate_subset(inputs=training_data_scaled, num_cells_per_input=model_hparams['batch_size'], weights=None, batch_size=1, return_indices=True)
            real_batch = np.squeeze(real_batch)
            labels_class_similarity = training_prior[indices_batch]

            #######################
            # update model
            _ = sess.run([optimize_model], feed_dict={data: real_batch, usc.class_similarity : labels_class_similarity})

            temp_assignments, temp_loss_model, temp_loss_reconst_data, temp_loss_kl_divergence_code_standard_gaussian, temp_loss_clustering_usc = sess.run([assignments, loss_model, loss_reconst_data, loss_kl_divergence_code_standard_gaussian, loss_clustering_usc], feed_dict={data: real_batch, usc.class_similarity : labels_class_similarity})

            if iteration % 10 == 0:
                f_measure = compute_f_measure(training_labels[indices_batch], temp_assignments)
                nmi = normalized_mutual_info_score(training_labels[indices_batch], temp_assignments)

            if np.isnan(temp_loss_model):
                raise Exception('NaN error in loss! -.-')

            # save loss plots
            list_loss_model.append(temp_loss_model)
            save_loss_plot(dir_output, list_loss_model, tag='_model')
            list_loss_reconst_data.append(temp_loss_reconst_data)
            save_loss_plot(dir_output, list_loss_reconst_data, tag='_reconst_data')
            list_loss_kl_div_standard_gaussian.append(temp_loss_kl_divergence_code_standard_gaussian)
            save_loss_plot(dir_output, list_loss_kl_div_standard_gaussian, tag='_kl_divergence_code_standard_gaussian')
            list_loss_clustering_usc.append(temp_loss_clustering_usc)
            save_loss_plot(dir_output, list_loss_clustering_usc, tag='_clustering_usc')

            if iteration == 0:
                best_loss = temp_loss_model
                saver.save(sess, os.path.join(dir_output, 'model.ckpt'))
            else:
                if (temp_loss_model < best_loss):
                    best_loss = temp_loss_model
                    saver.save(sess, os.path.join(dir_output, 'model.ckpt'))

            if model_hparams['plotting']:
                if iteration % 10 == 0:
                    temp_ind_test = np.random.choice(range(len(test_data_scaled)), model_hparams['batch_size_test'], replace=False)

                    temp_gates, temp_code, temp_gen_output_gated, temp_assignments, temp_centroids = sess.run([gates_assignments, code, gen_output_gated, assignments, centroids],feed_dict={data: test_data_scaled[temp_ind_test]})

                    if not os.path.exists(os.path.join(dir_output, str(iteration))):
                        os.makedirs(os.path.join(dir_output, str(iteration)))

                    saver.restore(sess, os.path.join(dir_output, 'model.ckpt'))
                    expert_id = np.argmax(temp_gates, axis=1)

                    # plot frequency of experts
                    temp_dir_output = os.path.join(dir_output, str(iteration), 'experts')
                    if not os.path.exists(temp_dir_output):
                        os.makedirs(temp_dir_output)

                    expert_frequency = [list(expert_id).count(x) for x in range(model_hparams['num_experts'])]
                    expert_frequency = expert_frequency / np.sum(expert_frequency)
                    plt.figure()
                    plt.bar(range(model_hparams['num_experts']), expert_frequency, align='center')
                    plt.xticks(range(model_hparams['num_experts']),[str(x) for x in range(model_hparams['num_experts'])])
                    plt.xlabel('Expert ID')
                    plt.ylabel('Expert weight')
                    plt.tight_layout()
                    plt.savefig(os.path.join(temp_dir_output, 'barplot_expert_weights.pdf'))
                    plt.close()

                    for i, label in enumerate(test_labels_unique):
                        temp_ind = np.where(test_labels[temp_ind_test] == label)[0]
                        plt.figure()
                        temp_test_gates_arg_max = list(expert_id[temp_ind])
                        plt.bar(range(model_hparams['num_experts']), [list(temp_test_gates_arg_max).count(x) for x in range(model_hparams['num_experts'])], tick_label=range(model_hparams['num_experts']))
                        plt.xlabel('Expert ID')
                        plt.ylabel('Frequency selected')
                        plt.tight_layout()
                        plt.savefig(os.path.join(temp_dir_output, 'bar_gates_' + str(label) + '.pdf'))
                        plt.close()

                    # plot code
                    temp_dir_output = os.path.join(dir_output, str(iteration), 'code')
                    if not os.path.exists(temp_dir_output):
                        os.makedirs(temp_dir_output)

                    pca_code = PCA(n_components=2)
                    pca_code.fit(temp_code)
                    pca_code_transform = pca_code.transform(np.vstack([temp_code]))
                    pca_centroids_transform = pca_code.transform(temp_centroids)

                    plt.figure()
                    plt.subplot(121)
                    for i, sub in enumerate(test_labels_unique):
                        temp_ind = np.where(test_labels[temp_ind_test] == sub)[0]
                        plt.scatter(pca_code_transform[temp_ind, 0], pca_code_transform[temp_ind, 1], c=dict_colors[sub], marker=dict_marker[sub], s=15)
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.title('Labels')
                    plt.tight_layout()
                    plt.subplot(122)
                    for i, sub in enumerate(np.unique(expert_id)):
                        temp_ind = np.where(expert_id == sub)[0]
                        plt.scatter(pca_code_transform[temp_ind, 0], pca_code_transform[temp_ind, 1], c=dict_colors_experts[sub], marker=dict_marker_experts[sub], s=15, label=str(sub))
                    plt.scatter(pca_centroids_transform[:, 0], pca_centroids_transform[:, 1], c='red', marker='+', s=15)
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.title('Gating')
                    plt.tight_layout()
                    plt.savefig(os.path.join(temp_dir_output, 'code.pdf'))
                    plt.close()

                    readme_current_training['iteration'] = iteration
                    readme_current_training['f-measure'] = f_measure
                    readme_current_training['NMI'] = nmi
                    readme_current_training['best_loss'] = best_loss
                    write_readme_file(readme_current_training, dir_output, 'current_training_state.txt')


            dict_losses = OrderedDict()
            dict_losses['iteration'] = iteration
            dict_losses['best_loss'] = best_loss
            dict_losses['loss_model'] = temp_loss_model
            dict_losses['reconst_data'] = temp_loss_reconst_data
            dict_losses['kl_div_code_standard_gaussian'] = temp_loss_kl_divergence_code_standard_gaussian
            dict_losses['usc'] = temp_loss_clustering_usc
            dict_losses['f-measure'] = f_measure
            print(dict_losses)

        pd.DataFrame(list_loss_model).to_csv(os.path.join(dir_output, 'loss_model.csv'), index=False, header=False, sep=',')
        pd.DataFrame(list_loss_reconst_data).to_csv(os.path.join(dir_output, 'loss_reconst_data.csv'), index=False,header=False, sep=',')
        pd.DataFrame(list_loss_kl_div_standard_gaussian).to_csv(os.path.join(dir_output, 'loss_kl_code_standard_gaussian.csv'), index=False, header=False, sep=',')
        pd.DataFrame(list_loss_clustering_usc).to_csv(os.path.join(dir_output, 'loss_usc.csv'), index=False,header=False, sep=',')


        ################################
        # plot test data
        if not os.path.exists(os.path.join(dir_output, str(iteration))):
            os.makedirs(os.path.join(dir_output, str(iteration)))

        # restore model
        saver.restore(sess, os.path.join(dir_output, 'model.ckpt'))

        # predict on test data
        list_temp_gates = list()
        list_temp_code = list()
        list_temp_gen_output_gated = list()
        list_temp_assignments = list()
        for i in range(len(test_data_scaled)):
            temp_gates, temp_code, temp_gen_output_gated, temp_assignments, temp_centroids = sess.run([gates_assignments, code, gen_output_gated, assignments, centroids], feed_dict={data: test_data_scaled[i].reshape([1, model_hparams['num_markers']])})
            temp_gen_output_gated = sigmoid(temp_gen_output_gated)

            list_temp_gates.append(temp_gates)
            list_temp_code.append(temp_code)
            list_temp_gen_output_gated.append(temp_gen_output_gated)
            list_temp_assignments.append(temp_assignments)

        temp_gates = np.vstack(list_temp_gates)
        temp_code = np.vstack(list_temp_code)
        temp_gen_output_gated = np.vstack(list_temp_gen_output_gated)
        temp_assignments = np.vstack(list_temp_assignments).squeeze()
        temp_ind_test = np.random.choice(range(len(test_data_scaled)), model_hparams['batch_size_test'], replace=False)
        temp_ind_test_gen = np.random.choice(range(len(temp_gen_output_gated)), model_hparams['batch_size_test'], replace=False)

        pd.DataFrame(temp_code).to_csv(os.path.join(dir_output, 'code.csv'), index=False, header=False, sep=',')
        pd.DataFrame(temp_assignments).to_csv(os.path.join(dir_output, 'assignments.csv'), index=False, header=False,sep=',')
        pd.DataFrame(temp_centroids).to_csv(os.path.join(dir_output, 'centroids.csv'), index=False, header=False, sep=',')

        # reconstruct data from moments predicted from gating
        expert_id = np.argmax(temp_gates, axis=1)

        temp_dir_output = os.path.join(dir_output, str(iteration))
        if not os.path.exists(temp_dir_output):
            os.makedirs(temp_dir_output)

        # plot frequency of experts
        temp_dir_output = os.path.join(dir_output, str(iteration), 'experts')
        if not os.path.exists(temp_dir_output):
            os.makedirs(temp_dir_output)

        expert_frequency = [list(expert_id).count(x) for x in range(model_hparams['num_experts'])]
        expert_frequency = expert_frequency / np.sum(expert_frequency)
        plt.figure()
        plt.bar(range(model_hparams['num_experts']), expert_frequency, align='center')
        plt.xticks(range(model_hparams['num_experts']), [str(x) for x in range(model_hparams['num_experts'])])
        plt.xlabel('Expert ID')
        plt.ylabel('Expert weight')
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir_output, 'barplot_expert_weights.pdf'))
        plt.close()

        for i, label in enumerate(test_labels_unique):
            temp_ind = np.where(test_labels == label)[0]
            plt.figure()
            temp_test_gates_arg_max = list(expert_id[temp_ind])
            plt.bar(range(model_hparams['num_experts']), [list(temp_test_gates_arg_max).count(x) for x in range(model_hparams['num_experts'])], tick_label=range(model_hparams['num_experts']))
            plt.xlabel('Expert ID')
            plt.ylabel('Frequency selected')
            plt.tight_layout()
            plt.savefig(os.path.join(temp_dir_output, 'bar_gates_' + str(label) + '.pdf'))
            plt.close()

        temp_dir_output = os.path.join(dir_output, str(iteration))
        if not os.path.exists(temp_dir_output):
            os.makedirs(temp_dir_output)

        # generator output
        temp_dir_output = os.path.join(dir_output, str(iteration), 'gen_output')
        if not os.path.exists(temp_dir_output):
            os.makedirs(temp_dir_output)

        pca_transformed = pca.transform(np.vstack([test_data_scaled[temp_ind_test], temp_gen_output_gated[temp_ind_test_gen]]))
        plt.figure()
        for i, sub in enumerate(test_labels_unique):
            temp_ind = np.where(test_labels[temp_ind_test] == sub)[0]
            plt.scatter(pca_transformed[temp_ind, 0], pca_transformed[temp_ind, 1], c=dict_colors[sub].reshape((1, 4)), s=1)
        for i, exp_id in enumerate(range(model_hparams['batch_size_test'])):
            temp_ind = np.where(expert_id[temp_ind_test_gen] == exp_id)[0]
            temp_ind += model_hparams['batch_size_test']
            for j in temp_ind:
                plt.text(pca_transformed[j, 0], pca_transformed[j, 1], str(exp_id), color='r', fontsize=6)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir_output, 'pca_all-real_vs_expert_gen_output.pdf'))
        plt.close()

        for id in np.unique(expert_id):
            temp_ind_id = np.where(expert_id == id)[0]
            temp_exp_ids = expert_id[temp_ind_id]

            pca_transformed = pca.transform(np.vstack([test_data_scaled[temp_ind_test], temp_gen_output_gated[temp_ind_id]]))
            plt.figure()
            for i, sub in enumerate(test_labels_unique):
                temp_ind = np.where(test_labels[temp_ind_test] == sub)[0]
                plt.scatter(pca_transformed[temp_ind, 0], pca_transformed[temp_ind, 1], c=dict_colors[sub].reshape((1, 4)), s=1)
            for j_id, j_sample in enumerate(range(model_hparams['batch_size_test'], pca_transformed.shape[0])):
                plt.text(pca_transformed[j_sample, 0], pca_transformed[j_sample, 1], temp_exp_ids[j_id], color='r', fontsize=6)
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.tight_layout()
            plt.savefig(os.path.join(temp_dir_output, str(id) + '_pca_all-real_vs_expert_gen_outputs.pdf'))
            plt.close()

        # plot code
        temp_dir_output = os.path.join(dir_output, str(iteration), 'code')
        if not os.path.exists(temp_dir_output):
            os.makedirs(temp_dir_output)

        pca_code = PCA()
        pca_code.fit(temp_code)
        pca_code_transform = pca_code.transform(np.vstack([temp_code]))
        pca_centroids_transform = pca_code.transform(temp_centroids)

        plt.figure()
        plt.subplot(121)
        for i, sub in enumerate(test_labels_unique):
            temp_ind = np.where(test_labels == sub)[0]
            plt.scatter(pca_code_transform[temp_ind, 0], pca_code_transform[temp_ind, 1], c=dict_colors[sub], marker=dict_marker[sub], s=15)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Labels')
        plt.tight_layout()
        plt.subplot(122)
        for i, sub in enumerate(np.unique(expert_id)):
            temp_ind = np.where(expert_id == sub)[0]
            plt.scatter(pca_code_transform[temp_ind, 0], pca_code_transform[temp_ind, 1], c=dict_colors_experts[sub], marker=dict_marker_experts[sub], s=15, label=str(sub))
        plt.scatter(pca_centroids_transform[:, 0], pca_centroids_transform[:, 1], c='red', marker='+', s=15)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Gating')
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir_output, 'code.pdf'))
        plt.close()


        f_measure = compute_f_measure(test_labels, expert_id)
        model_hparams['f-measure'] = f_measure
        model_hparams['NMI'] = normalized_mutual_info_score(test_labels, expert_id)
        model_hparams['best_loss'] = best_loss
        write_readme_file(model_hparams, dir_output)


if __name__ == '__main__':
    main()

