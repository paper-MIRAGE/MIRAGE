"""
Script for training the Mirage model
"""
import json
import math
import os
import sys
import timeit
import uuid
from datetime import datetime
from pathlib import Path
from pickle import dump

import numpy as np
import sacred
import sklearn
from sacred.stflow import LogFileWriter
from sklearn import metrics
from tqdm import tqdm

from model import Mirage
from utils import batch_generator, get_data

try:
    import tensorflow.compat.v1 as tf

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

tf.set_random_seed(24)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

ex = sacred.Experiment("mirage")
ex.observers.append(
    sacred.observers.FileStorageObserver.create("../sacred_runs_players_counters")
)
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds
ex.add_config("../config.json")

with open("../config.json", "r") as f:
    config_dict = json.load(f)


@ex.config
def ex_config():
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_{}_{}".format(
        name,
        datetime.now().strftime("%Y-%m-%d-%H%M%S"),
        uuid.uuid4().hex[:5],
    )
    logdir = "../logs/{}".format(ex_name)
    loss_logdir = "./logs/loss/{}".format(ex_name)
    modelpath = "../models/{}/{}".format(ex_name, ex_name)
    path = Path(config_dict["checkpoint_dir"])
    path.mkdir(parents=True, exist_ok=True)


@ex.capture
def train_model(
    model,
    data_train,
    data_val,
    endpoints_total_train,
    endpoints_total_val,
    data_train_full,
    endpoints_total_train_full,
    data_val_full,
    endpoints_total_val_full,
    lr_val,
    prior_val,
    num_epochs,
    batch_size,
    latent_dim,
    som_dim,
    learning_rate,
    epochs_pretrain,
    epochs_forecasting_finetuning,
    ex_name,
    logdir,
    loss_logdir,
    modelpath,
    val_epochs,
    evaluate_mse_flag,
    evaluate_nmi_flag,
    label_index_for_nmi,
    save_pretrain,
    use_saved_pretrain,
    benchmark,
    train_ratio,
    annealtime,
    lstm_dim,
    c_dim,
    trans_mat_size,
    max_n_step,
    num_pred,
    epochs_prediction_finetuning,
    checkpoint_dir,
):
    """Trains the Mirage model.
    Params:
        model (Mirage): Mirage model to train.
        data_train (np.array): Training set.
        data_val (np.array): Validation/test set.
        endpoints_total_val (np.array): Validation/test labels.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for the training.
        latent_dim (int): Dimensionality of the Mirage's latent space.
        som_dim (list): Dimensionality of the self-organizing map.
        learning_rate (float): Learning rate for the optimization.
        epochs_pretrain (int): Number of VAE pretraining epochs.
        epochs_forecasting_finetuning (int): Number of Forecasting finetuning epochs.
        ex_name (string): Unique name of this particular run.
        logdir (path): Directory for the experiment logs.
        loss_logdir (path): Directory for the Losses logs pickle files.
        modelpath (path): Path for the model checkpoints.
        val_epochs (bool): If "True" clustering results are saved every 10 epochs on default output files.
    Steps:
        1. Transition Pretraining
        2. Autoencoder Pretraining
        3. SOM Initialization
        4. Full Training
        5. Prediction Fine-tune
    """
    epochs = 0
    iterations = 0
    pretrainpath = "../models/pretrain/LSTM"
    len_data_train = len(data_train)
    len_data_val = len(data_val)
    num_batches = len_data_train // batch_size
    print("Num Batches:", num_batches)
    print("Data Train Shape:", data_train.shape)
    print("Data Val Shape:", data_val.shape)

    train_gen = batch_generator(
        data_train,
        endpoints_total_train,
        data_val,
        endpoints_total_val,
        batch_size,
        mode="train",
    )
    val_gen = batch_generator(
        data_train,
        endpoints_total_train,
        data_val,
        endpoints_total_val,
        batch_size,
        mode="val",
    )

    saver = tf.train.Saver(max_to_keep=5)
    summaries = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with LogFileWriter(ex):
            train_writer = tf.summary.FileWriter(logdir + "/train", sess.graph)
            test_writer = tf.summary.FileWriter(logdir + "/test", sess.graph)

        (
            train_step_total,
            train_step_ae,
            train_step_som,
            train_step_pred,
            train_step_trans,
            train_step_forecast,
        ) = model.optimize
        x = model.inputs
        c = model.conditions
        x_next = model.x_next_inputs
        x_trans_mat = model.x_trans_mat_inputs
        y_trans_mat = model.y_trans_mat_inputs
        concat_x_c = model.concatenated_inputs
        p = model.p
        is_training = model.is_training
        graph = tf.get_default_graph()
        init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
        z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
        z_e_rec = graph.get_tensor_by_name("reconstruction_e/decoder/z_e:0")
        training_dic = {
            is_training: True,
            z_e_p: np.zeros((max_n_step, batch_size, latent_dim)),
            init_1: np.zeros((2, batch_size, lstm_dim)),
            z_e_rec: np.zeros((max_n_step * batch_size, latent_dim + c_dim)),
        }

        pbar = tqdm(total=(num_epochs + epochs_pretrain * 3) * (num_batches))

        print("\n********** Starting job {} ********* \n".format(ex_name))

        train_trans_loss_dict = {}
        val_trans_loss_dict = {}

        train_recons_loss_dict = {}
        val_recons_loss_dict = {}

        train_som_loss_a_dict = {}
        val_som_loss_a_dict = {}

        train_total_loss_dict = {}
        val_total_loss_dict = {}

        train_prediction_loss_dict = {}
        val_prediction_loss_dict = {}

        train_forecasting_loss_dict = {}
        val_forecasting_loss_dict = {}

        a = np.zeros((batch_size * max_n_step, som_dim[0] * som_dim[1]))
        dp = {p: a}
        dp.update(training_dic)

        if benchmark:
            ttime_per_epoch = []
            ttime_trans_per_epoch = []
            ttime_ae_per_epoch = []
            ttime_som_per_epoch = []
            ttime_pred_per_epoch = []
            ttime_forecast_per_epoch = []

        if use_saved_pretrain:
            print("\n\nUsing Saved Pretraining...\n")
            saver.restore(sess, pretrainpath)
        else:
            print("\n\nTransition Pretraining...\n")
            if benchmark:
                t_begin_all = timeit.default_timer()
            prior = 0
            for epoch in range(epochs_pretrain):
                if epoch > 10:
                    prior = min(prior + (1.0 / annealtime), 1.0)
                if benchmark:
                    t_begin = timeit.default_timer()
                for i in range(num_batches):
                    batch_data, batch_labels, ii = next(train_gen)

                    batch_ts_data = batch_data[:, :, trans_mat_size:]
                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                    batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]

                    f_dic = {
                        x: batch_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                    }

                    f_dic.update(dp)
                    train_step_trans.run(feed_dict=f_dic)
                    pbar.set_postfix(
                        epoch=epoch,
                        refresh=False,
                    )
                    pbar.update(1)
                if benchmark:
                    t_end = timeit.default_timer()
                    ttime_trans_per_epoch.append(t_end - t_begin)

            if benchmark:
                t_end_all = timeit.default_timer()
                ttime_trans_pretrain = t_end_all - t_begin_all

            print("\n\nAutoencoder Pretraining...\n")
            if benchmark:
                t_begin_all = timeit.default_timer()
            prior = 0
            for epoch in range(epochs_pretrain):
                if epoch > 10:
                    prior = min(prior + (1.0 / annealtime), 1.0)
                if benchmark:
                    t_begin = timeit.default_timer()
                for i in range(num_batches):
                    batch_data, batch_labels, ii = next(train_gen)

                    if np.any(np.isnan(batch_data)):
                        raise ValueError(
                            "Inside Autoencoder Pretraining "
                            f"Nan on batch {i} in epoch {epoch}"
                        )

                    batch_ts_data = batch_data[:, :, trans_mat_size:]
                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                    batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]

                    f_dic = {
                        x: batch_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                        lr_val: learning_rate,
                        prior_val: prior,
                    }

                    f_dic.update(dp)
                    train_step_ae.run(feed_dict=f_dic)

                    pbar.set_postfix(
                        epoch=epoch,
                        refresh=False,
                    )
                    pbar.update(1)
                if benchmark:
                    t_end = timeit.default_timer()
                    ttime_ae_per_epoch.append(t_end - t_begin)

            if benchmark:
                t_end_all = timeit.default_timer()
                ttime_ae_pretrain = t_end_all - t_begin_all

            print("\n\nSOM initialization...\n")
            if benchmark:
                t_begin_all = timeit.default_timer()

            for epoch in range(epochs_pretrain // 3):
                if benchmark:
                    t_begin = timeit.default_timer()
                for i in range(num_batches):
                    batch_data, batch_labels, ii = next(train_gen)

                    batch_ts_data = batch_data[:, :, trans_mat_size:]
                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                    batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]

                    f_dic = {
                        x: batch_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                        lr_val: 0.1,
                    }

                    f_dic.update(dp)
                    train_step_som.run(feed_dict=f_dic)
                    pbar.set_postfix(
                        epoch=epoch,
                        refresh=False,
                    )
                    pbar.update(1)
                if benchmark:
                    t_end = timeit.default_timer()
                    ttime_som_per_epoch.append(t_end - t_begin)

            for epoch in range(epochs_pretrain // 3):
                if benchmark:
                    t_begin = timeit.default_timer()
                for i in range(num_batches):
                    batch_data, batch_labels, ii = next(train_gen)

                    batch_ts_data = batch_data[:, :, trans_mat_size:]
                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                    batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]

                    f_dic = {
                        x: batch_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                        lr_val: 0.01,
                    }

                    f_dic.update(dp)
                    train_step_som.run(feed_dict=f_dic)

                    pbar.set_postfix(
                        epoch=epoch,
                        refresh=False,
                    )
                    pbar.update(1)
                if benchmark:
                    t_end = timeit.default_timer()
                    ttime_som_per_epoch.append(t_end - t_begin)

            for epoch in range(epochs_pretrain // 3):
                if benchmark:
                    t_begin = timeit.default_timer()
                for i in range(num_batches):
                    batch_data, batch_labels, ii = next(train_gen)

                    batch_ts_data = batch_data[:, :, trans_mat_size:]
                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                    batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]

                    f_dic = {
                        x: batch_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                        lr_val: 0.001,
                    }
                    f_dic.update(dp)
                    train_step_som.run(feed_dict=f_dic)
                    pbar.set_postfix(
                        epoch=epoch,
                        refresh=False,
                    )
                    pbar.update(1)
                if benchmark:
                    t_end = timeit.default_timer()
                    ttime_som_per_epoch.append(t_end - t_begin)

            if benchmark:
                t_end_all = timeit.default_timer()
                ttime_som = t_end_all - t_begin_all

            if save_pretrain:
                saver.save(sess, pretrainpath)

        print("\n\nTraining...\n")

        if benchmark:
            t_begin_all = timeit.default_timer()

        prior = 0
        for epoch in range(num_epochs):
            if epoch > 10:
                prior = min(prior + (1.0 / annealtime), 1.0)
            if benchmark:
                t_begin = timeit.default_timer()
            epochs += 1

            ###############################################################
            #       Generate Target Distribution for Training set         #
            ###############################################################
            f_dic = {x: data_train}
            f_dic.update(training_dic)
            q = []
            blocks_size = 19  # nothing to do with the number of time windows.
            for t in range(blocks_size):
                batch_ts_data = data_train[
                    int(len(data_train) / (blocks_size + 1))
                    * t : int(len(data_train) / (blocks_size + 1))
                    * (t + 1),
                    :,
                    trans_mat_size:,
                ]
                batch_conditions = np.zeros((batch_ts_data.shape[0], max_n_step, c_dim))
                batch_concat_x_c = np.concatenate(
                    (batch_ts_data, batch_conditions), axis=2
                )
                f_dic = {
                    x: batch_ts_data,
                    x_trans_mat: data_train[
                        int(len(data_train) / (blocks_size + 1))
                        * t : int(len(data_train) / (blocks_size + 1))
                        * (t + 1),
                        :,
                        :trans_mat_size,
                    ],
                    c: batch_conditions,
                    concat_x_c: batch_concat_x_c,
                }

                batch_q = sess.run(model.q, feed_dict=f_dic)
                q.extend(batch_q)

            batch_ts_data = data_train[
                int(len(data_train) / (blocks_size + 1)) * blocks_size :,
                :,
                trans_mat_size:,
            ]
            batch_conditions = np.zeros(
                (
                    len(
                        data_train[
                            int(len(data_train) / (blocks_size + 1)) * blocks_size :,
                            :,
                            trans_mat_size:,
                        ]
                    ),
                    max_n_step,
                    c_dim,
                )
            )
            batch_concat_x_c = np.concatenate((batch_ts_data, batch_conditions), axis=2)

            q.extend(
                sess.run(
                    model.q,
                    feed_dict={
                        x: batch_ts_data,
                        x_trans_mat: data_train[
                            int(len(data_train) / (blocks_size + 1)) * blocks_size :,
                            :,
                            :trans_mat_size,
                        ],
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                    },
                )
            )
            q = np.array(q)
            ppt = model.target_distribution(q)

            ###############################################################
            #       Generate Target Distribution for Validation set       #
            ###############################################################
            q = []
            f_dic = {x: data_val}
            f_dic.update(training_dic)
            blocks_size = 9  # nothing has do with length of obeservation per player.
            for t in range(blocks_size):
                batch_ts_data = data_val[
                    int(len(data_val) / (blocks_size + 1))
                    * t : int(len(data_val) / (blocks_size + 1))
                    * (t + 1),
                    :,
                    trans_mat_size:,
                ]
                batch_conditions = np.zeros(
                    (
                        (int(len(data_val) / (blocks_size + 1)) * (t + 1))
                        - (int(len(data_val) / (blocks_size + 1)) * t),
                        max_n_step,
                        c_dim,
                    )
                )
                batch_concat_x_c = np.concatenate(
                    (batch_ts_data, batch_conditions), axis=2
                )

                f_dic = {
                    x: batch_ts_data,
                    x_trans_mat: data_val[
                        int(len(data_val) / (blocks_size + 1))
                        * t : int(len(data_val) / (blocks_size + 1))
                        * (t + 1),
                        :,
                        :trans_mat_size,
                    ],
                    c: batch_conditions,
                    concat_x_c: batch_concat_x_c,
                }
                batch_q = sess.run(model.q, feed_dict=f_dic)
                q.extend(batch_q)

            batch_ts_data = data_val[
                int(len(data_val) / (blocks_size + 1)) * blocks_size :,
                :,
                trans_mat_size:,
            ]
            batch_conditions = np.zeros(
                (
                    len(
                        data_val[
                            int(len(data_val) / (blocks_size + 1)) * blocks_size :,
                            :,
                            trans_mat_size:,
                        ]
                    ),
                    max_n_step,
                    c_dim,
                )
            )
            batch_concat_x_c = np.concatenate((batch_ts_data, batch_conditions), axis=2)
            q.extend(
                sess.run(
                    model.q,
                    feed_dict={
                        x: batch_ts_data,
                        x_trans_mat: data_val[
                            int(len(data_val) / (blocks_size + 1)) * blocks_size :,
                            :,
                            :trans_mat_size,
                        ],
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                    },
                )
            )
            q = np.array(q)

            for i in range(num_batches):
                iterations += 1
                #####################
                #    Training set   #
                #####################
                batch_data, batch_labels, ii = next(train_gen)

                batch_ts_data = batch_data[:, :, trans_mat_size:]
                batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]
                batch_conditions = np.zeros((batch_data.shape[0], max_n_step, c_dim))
                batch_concat_x_c = np.concatenate(
                    (batch_ts_data, batch_conditions), axis=2
                )

                ftrain = {
                    p: ppt[
                        ii
                        * batch_size
                        * max_n_step : (ii + 1)
                        * batch_size
                        * max_n_step
                    ]
                }
                f_dic = {
                    x: batch_ts_data,
                    x_trans_mat: batch_trans_mat_x,
                    y_trans_mat: batch_trans_mat_y,
                    lr_val: learning_rate,
                    prior_val: prior,
                    c: batch_conditions,
                    concat_x_c: batch_concat_x_c,
                }

                f_dic.update(ftrain)
                f_dic.update(training_dic)
                train_step_total.run(feed_dict=f_dic)
                train_step_pred.run(feed_dict=f_dic)

                pbar.set_postfix(epoch=epoch, refresh=False)
                pbar.update(1)

            if val_epochs == True and epoch % 5 == 0:
                exp_name = modelpath.split("/")[2]
                chkpt_base_dir = checkpoint_dir + exp_name + "/"
                cur_path = chkpt_base_dir + "chkpt_at_" + str(epoch)
                saver.save(sess, cur_path)

                # For MSE on Reconstruction
                results_mse = evaluate_reconstruction_mse(
                    model,
                    data_train_full,
                    endpoints_total_train_full,
                    data_val_full,
                    endpoints_total_val_full,
                    x,
                    x_trans_mat,
                    c_dim,
                    modelpath,
                    trans_mat_size,
                    c,
                    concat_x_c,
                    max_n_step,
                )

                print(results_mse)

            if benchmark:
                t_end = timeit.default_timer()
                ttime_per_epoch.append(t_end - t_begin)

        if benchmark:
            t_end_all = timeit.default_timer()
            ttime_training = t_end_all - t_begin_all

        print("\n\nPrediction Finetuning...\n")
        if benchmark:
            t_begin_all = timeit.default_timer()

        for epoch in range(epochs_prediction_finetuning):  # 50 epochs
            if benchmark:
                t_begin = timeit.default_timer()
            for i in range(num_batches):
                batch_data, batch_labels, ii = next(train_gen)

                batch_ts_data = batch_data[:, :, trans_mat_size:]
                batch_trans_mat_x = batch_data[:, :, :trans_mat_size]
                batch_trans_mat_y = batch_labels[:, :, :trans_mat_size]
                batch_conditions = np.zeros((batch_data.shape[0], max_n_step, c_dim))
                batch_concat_x_c = np.concatenate(
                    (batch_ts_data, batch_conditions), axis=2
                )
                f_dic = {
                    x: batch_ts_data,
                    x_trans_mat: batch_trans_mat_x,
                    y_trans_mat: batch_trans_mat_y,
                    lr_val: learning_rate,
                    prior_val: prior,
                    c: batch_conditions,
                    concat_x_c: batch_concat_x_c,
                }
                f_dic.update(dp)
                train_step_pred.run(feed_dict=f_dic)

                if i % 100 == 0:
                    #####################
                    #  Validation set   #
                    #####################
                    batch_val_data, batch_val_labels, ii = next(val_gen)

                    batch_val_ts_data = batch_val_data[:, :, trans_mat_size:]
                    batch_val_trans_mat_x = batch_val_data[:, :, :trans_mat_size]
                    batch_val_trans_mat_y = batch_val_labels[:, :, :trans_mat_size]
                    batch_conditions = np.zeros(
                        (batch_val_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_val_ts_data, batch_conditions), axis=2
                    )

                    f_dic = {
                        x: batch_val_ts_data,
                        x_trans_mat: batch_val_trans_mat_x,
                        y_trans_mat: batch_val_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                    }
                    f_dic.update(dp)

                    test_loss, summary = sess.run(
                        [model.loss_prediction, summaries], feed_dict=f_dic
                    )
                    val_prediction_loss_dict[epoch * num_batches + i] = test_loss
                    test_writer.add_summary(
                        summary, tf.train.global_step(sess, model.global_step)
                    )

                    #####################
                    #    Training set   #
                    #####################

                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    f_dic = {
                        x: batch_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                    }
                    f_dic.update(dp)

                    train_loss, summary = sess.run(
                        [model.loss_prediction, summaries], feed_dict=f_dic
                    )
                    if math.isnan(train_loss):
                        print(
                            "Inside Prediciton finetuning Nan on batch",
                            i,
                            "in epoch",
                            epoch,
                        )
                        return None
                    train_prediction_loss_dict[epoch * num_batches + i] = train_loss
                    train_writer.add_summary(
                        summary, tf.train.global_step(sess, model.global_step)
                    )
                pbar.set_postfix(
                    epoch=epoch,
                    train_loss=train_loss,
                    test_loss=test_loss,
                    refresh=False,
                )
                pbar.update(1)

            if val_epochs == True and epoch % 5 == 0:
                exp_name = modelpath.split("/")[2]

                chkpt_base_dir = checkpoint_dir + exp_name + "/"
                cur_path = chkpt_base_dir + "chkpt_at_" + str(epoch)
                saver.save(sess, cur_path)

                # For MSE on Reconstruction
                results_mse = evaluate_reconstruction_mse(
                    model,
                    data_train_full,
                    endpoints_total_train_full,
                    data_val_full,
                    endpoints_total_val_full,
                    x,
                    x_trans_mat,
                    c_dim,
                    modelpath,
                    trans_mat_size,
                    c,
                    concat_x_c,
                    max_n_step,
                )

                print(results_mse)

            if benchmark:
                t_end = timeit.default_timer()
                ttime_pred_per_epoch.append(t_end - t_begin)

        if benchmark:
            t_end_all = timeit.default_timer()
            ttime_pred = t_end_all - t_begin_all

        train_full_gen = batch_generator(
            data_train_full,
            endpoints_total_train_full,
            data_val_full,
            endpoints_total_val_full,
            batch_size,
            mode="train",
        )
        val_full_gen = batch_generator(
            data_train_full,
            endpoints_total_train_full,
            data_val_full,
            endpoints_total_val_full,
            batch_size,
            mode="val",
        )
        print("\n\nForecasting Finetuning...\n")
        if benchmark:
            t_begin_all = timeit.default_timer()

        for epoch in range(epochs_forecasting_finetuning):
            if benchmark:
                t_begin = timeit.default_timer()

            for i in range(num_batches):
                batch_data, batch_labels, ii = next(train_full_gen)
                for j in range(num_pred - 1):
                    batch_ts_data = batch_data[:, j : max_n_step + j, trans_mat_size:]
                    batch_trans_mat_x = batch_data[
                        :, j : max_n_step + j, :trans_mat_size
                    ]
                    batch_trans_mat_y = batch_labels[
                        :, j : max_n_step + j, :trans_mat_size
                    ]
                    batch_conditions = np.zeros(
                        (batch_data.shape[0], max_n_step, c_dim)
                    )
                    batch_concat_x_c = np.concatenate(
                        (batch_ts_data, batch_conditions), axis=2
                    )

                    batch_next_ts_data = batch_data[
                        :, j + 1 : max_n_step + j + 1, trans_mat_size:
                    ]
                    f_dic = {
                        x: batch_ts_data,
                        x_next: batch_next_ts_data,
                        x_trans_mat: batch_trans_mat_x,
                        y_trans_mat: batch_trans_mat_y,
                        lr_val: learning_rate,
                        prior_val: prior,
                        c: batch_conditions,
                        concat_x_c: batch_concat_x_c,
                    }
                    f_dic.update(dp)
                    train_step_forecast.run(feed_dict=f_dic)

                pbar.set_postfix(
                    epoch=epoch,
                    refresh=False,
                )
                pbar.update(1)

            if val_epochs == True and epoch % 5 == 0:
                exp_name = modelpath.split("/")[2]

                chkpt_base_dir = checkpoint_dir + exp_name + "/"
                cur_path = chkpt_base_dir + "chkpt_at_" + str(epoch)
                saver.save(sess, cur_path)

                # For MSE on Reconstruction
                results_mse = evaluate_reconstruction_mse(
                    model,
                    data_train_full,
                    endpoints_total_train_full,
                    data_val_full,
                    endpoints_total_val_full,
                    x,
                    x_trans_mat,
                    c_dim,
                    modelpath,
                    trans_mat_size,
                    c,
                    concat_x_c,
                    max_n_step,
                )

                print(results_mse)

            if benchmark:
                t_end = timeit.default_timer()
                ttime_forecast_per_epoch.append(t_end - t_begin)

        if benchmark:
            t_end_all = timeit.default_timer()
            ttime_forecast = t_end_all - t_begin_all

        # Save loss dict into a pickle file.
        print("loss_logdir:", loss_logdir[2:])
        os.makedirs(loss_logdir[2:], exist_ok=True)

        with open(loss_logdir + "/train_trans_loss.pkl", "wb") as file:
            dump(train_trans_loss_dict, file)
        with open(loss_logdir + "/val_trans_loss.pkl", "wb") as file:
            dump(val_trans_loss_dict, file)

        with open(loss_logdir + "/train_recons_loss.pkl", "wb") as file:
            dump(train_recons_loss_dict, file)
        with open(loss_logdir + "/val_recons_loss.pkl", "wb") as file:
            dump(val_recons_loss_dict, file)

        with open(loss_logdir + "/train_som_loss_a.pkl", "wb") as file:
            dump(train_som_loss_a_dict, file)
        with open(loss_logdir + "/val_som_loss_a.pkl", "wb") as file:
            dump(val_som_loss_a_dict, file)

        with open(loss_logdir + "/train_total_loss.pkl", "wb") as file:
            dump(train_total_loss_dict, file)
        with open(loss_logdir + "/val_total_loss.pkl", "wb") as file:
            dump(val_total_loss_dict, file)

        with open(loss_logdir + "/train_prediction_loss.pkl", "wb") as file:
            dump(train_prediction_loss_dict, file)
        with open(loss_logdir + "/val_prediction_loss.pkl", "wb") as file:
            dump(val_prediction_loss_dict, file)

        with open(loss_logdir + "/train_forecasting_loss.pkl", "wb") as file:
            dump(train_forecasting_loss_dict, file)
        with open(loss_logdir + "/val_forecasting_loss.pkl", "wb") as file:
            dump(val_forecasting_loss_dict, file)

        saver.save(sess, modelpath)
        final_results = {}

        if evaluate_mse_flag:
            results_mse = evaluate_reconstruction_mse(
                model,
                data_train_full,
                endpoints_total_train_full,
                data_val_full,
                endpoints_total_val_full,
                x,
                x_trans_mat,
                c_dim,
                modelpath,
                trans_mat_size,
                c,
                concat_x_c,
                max_n_step,
            )
            final_results.update(results_mse)

        if evaluate_nmi_flag:
            results_nmi = evaluate_nmi(
                model,
                label_index_for_nmi,
                x,
                x_trans_mat,
                y_trans_mat,
                c_dim,
                val_gen,
                len_data_val,
                modelpath,
                epochs,
                trans_mat_size,
                c,
                concat_x_c,
                max_n_step,
            )
            final_results.update(results_nmi)

        pbar.close()

        if benchmark:
            print(
                "\nNumber of time series in train: {} %, {}".format(
                    train_ratio, len(data_train)
                )
            )
            print("Trans pretrain time: {:.3f}".format(ttime_trans_pretrain))
            print(
                "Trans pretrain time per epoch: {:.3f}".format(
                    np.mean(ttime_ae_per_epoch)
                )
            )
            print("AE pretrain time: {:.3f}".format(ttime_ae_pretrain))
            print(
                "AE pretrain time per epoch: {:.3f}".format(np.mean(ttime_ae_per_epoch))
            )
            print("SOM init time: {:.3f}".format(ttime_som))
            print(
                "SOM init time per epoch: {:.3f}".format(np.mean(ttime_som_per_epoch))
            )
            print("Training time: {:.3f}".format(ttime_training))
            print("Training time per epoch: {:.3f}".format(np.mean(ttime_per_epoch)))
            print("Pred finetuning time: {:.3f}".format(ttime_pred))
            print(
                "Pred finetuning time per epoch: {:.3f}".format(
                    np.mean(ttime_pred_per_epoch)
                )
            )
            print("Forecast finetuning time: {:.3f}".format(ttime_forecast))
            print(
                "Forecast finetuning time per epoch: {:.3f}".format(
                    np.mean(ttime_forecast_per_epoch)
                )
            )
            sys.exit(0)

        return final_results


@ex.capture
def evaluate_reconstruction_mse(
    model,
    data_train_full,
    endpoints_total_train_full,
    data_val_full,
    endpoints_total_val_full,
    x,
    x_trans_mat,
    c_dim,
    modelpath,
    trans_mat_size,
    c,
    concat_x_c,
    max_n_step,
    num_pred,
    batch_size,
    latent_dim,
    lstm_dim,
    input_size,
    ex_name,
):
    data_train = data_train_full[:, :max_n_step, :]
    endpoints_total_train = endpoints_total_train_full[:, :max_n_step, :]

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0)
    val_gen = batch_generator(
        data_train,
        endpoints_total_train,
        data_val_full,
        endpoints_total_val_full,
        batch_size,
        mode="val",
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)
        is_training = model.is_training

        graph = tf.get_default_graph()
        next_z_e = graph.get_tensor_by_name("prediction/next_z_e:0")
        x = graph.get_tensor_by_name("inputs/x:0")
        init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
        z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
        state1 = graph.get_tensor_by_name("prediction/next_state/next_state:0")
        embeddings = graph.get_tensor_by_name("embeddings/embeddings:0")
        z_p = graph.get_tensor_by_name("reconstruction_e/decoder/z_e:0")
        reconstruction = graph.get_tensor_by_name("reconstruction_e/x_hat:0")
        c = graph.get_tensor_by_name("conditions/c:0")
        concat_x_c = graph.get_tensor_by_name("concatenated_inputs/concat_x_c:0")
        trans_mat_op = graph.get_tensor_by_name("shift_x_trans_mat/trans_mat_op:0")
        conditions_op = graph.get_tensor_by_name(
            "condition_pred_y/conditions_one_hot:0"
        )

        #####################################################################################
        #   I. Generate embeddings for validation data till (max_n_step - num_pred) step    #
        #####################################################################################

        num_batches = len(data_val_full) // batch_size
        total_val_size = num_batches * batch_size
        print("len(data_val_full)", len(data_val_full))
        print("total_val_size", total_val_size)

        training_dic = {
            is_training: True,
            z_e_p: np.zeros((max_n_step, len(data_val_full), latent_dim)),
            init_1: np.zeros((2, batch_size, lstm_dim)),
            z_p: np.zeros((max_n_step * len(data_val_full), latent_dim + c_dim)),
        }

        embeddings = sess.run(
            model.embeddings,
            feed_dict={x: data_val_full[:total_val_size, :max_n_step, trans_mat_size:]},
        )
        embeddings = np.reshape(embeddings, (-1, latent_dim))

        next_z_e_o = []
        next_trans_o = []
        state1_o = []
        total = 0
        for i in range(num_batches):
            batch_val_data, _, ii = next(val_gen)
            batch_val_data = batch_val_data[:, :max_n_step, :]
            total += len(batch_val_data)

            # Added for CVAE
            batch_val_ts_data = batch_val_data[:, :, trans_mat_size:]
            batch_val_trans_mat_x = batch_val_data[:, :, :trans_mat_size]
            batch_conditions = np.zeros((batch_val_ts_data.shape[0], max_n_step, c_dim))
            batch_concat_x_c = np.concatenate(
                (batch_val_ts_data, batch_conditions), axis=2
            )
            f_dic = {
                x: batch_val_ts_data,
                x_trans_mat: batch_val_trans_mat_x,
                c: batch_conditions,
                concat_x_c: batch_concat_x_c,
            }
            f_dic.update(training_dic)
            next_z_e_o.extend(sess.run(next_z_e, feed_dict=f_dic))
            next_trans_o.extend(sess.run(trans_mat_op, feed_dict=f_dic))
            if i == 0:
                state1_o = sess.run(state1, feed_dict=f_dic)
            else:
                state1_o = np.concatenate(
                    [state1_o, sess.run(state1, feed_dict=f_dic)], axis=1
                )
        next_z_e_o = np.array(next_z_e_o)

        state1_o = np.array(state1_o)
        next_z_e_o_all = np.reshape(next_z_e_o, (-1, max_n_step, latent_dim))
        next_z_e_o = np.reshape(next_z_e_o, (-1, latent_dim))

        next_trans_o = np.array(next_trans_o)
        next_trans_o = np.reshape(next_trans_o, (-1, max_n_step, trans_mat_size))

        next_cond_o = sess.run(conditions_op, feed_dict={x_trans_mat: next_trans_o})

        batch_conditions = np.zeros((total_val_size, 1, c_dim))
        batch_val_ts_data = np.zeros((total_val_size, 1, input_size))
        batch_concat_x_c = np.concatenate((batch_val_ts_data, batch_conditions), axis=2)

        f_dic = {
            x: batch_val_ts_data,
            x_trans_mat: np.zeros((total_val_size, 1, trans_mat_size)),
            is_training: False,
            z_e_p: np.zeros((total_val_size, 1, latent_dim)),
            z_p: np.concatenate([next_z_e_o, next_cond_o], axis=-1),
            init_1: np.zeros((2, batch_size, lstm_dim)),
            c: batch_conditions,
            concat_x_c: batch_concat_x_c,
        }
        x_pred_hat = np.reshape(
            sess.run(reconstruction, feed_dict=f_dic), (-1, max_n_step, input_size)
        )

        next_trans_o = np.reshape(
            next_trans_o[:, -1, :], (total_val_size, 1, trans_mat_size)
        )
        next_z_e_o_for_pred = np.reshape(next_z_e_o, (-1, max_n_step, latent_dim))

        ###########################################################################
        #  II. Updated pred with sliced data and embeddings for each timestamp.   #
        ###########################################################################

        for i in range(num_pred - 1):
            # --------------------#
            # Prediction Module  #
            # --------------------#
            inp = data_val_full[:total_val_size, i : (max_n_step + i), trans_mat_size:]
            inp_trans = data_val_full[
                :total_val_size, i : (max_n_step + i), :trans_mat_size
            ]

            batch_conditions = np.zeros((inp.shape[0], max_n_step, c_dim))
            batch_concat_x_c = np.concatenate((inp, batch_conditions), axis=2)

            f_dic = {x: inp, x_trans_mat: inp_trans}
            val_dic = {
                is_training: False,
                z_e_p: next_z_e_o_for_pred,
                init_1: state1_o,
                z_p: np.zeros((max_n_step * len(inp), latent_dim + c_dim)),
                c: batch_conditions,
                concat_x_c: batch_concat_x_c,
            }
            f_dic.update(val_dic)
            next_z_e_o = sess.run(next_z_e, feed_dict=f_dic)
            state1_o = sess.run(state1, feed_dict=f_dic)

            next_z_e_o_all = np.concatenate(
                [
                    next_z_e_o_all,
                    np.reshape(
                        next_z_e_o[:, -1, :],
                        (
                            next_z_e_o[:, -1, :].shape[0],
                            1,
                            next_z_e_o[:, -1, :].shape[1],
                        ),
                    ),
                ],
                axis=1,
            )
            next_z_e_o_for_pred = next_z_e_o_all[:, i + 1 :, :]

            # --------------------#
            # Transition Module  #
            # --------------------#
            f_dic = {x_trans_mat: next_trans_o}
            next_trans_o = sess.run(trans_mat_op, feed_dict=f_dic)
            next_trans_o = np.reshape(next_trans_o, (-1, 1, trans_mat_size))
            next_cond_o = sess.run(conditions_op, feed_dict={x_trans_mat: next_trans_o})

            # ------------------------#
            # Reconstruction Module  #
            # ------------------------#
            batch_val_ts_data = np.zeros((total_val_size, 1, input_size))
            batch_conditions = np.zeros((total_val_size, 1, c_dim))
            batch_concat_x_c = np.concatenate(
                (batch_val_ts_data, batch_conditions), axis=2
            )

            next_z_e_o = next_z_e_o[:, -1, :]
            f_dic = {
                x: batch_val_ts_data,
                x_trans_mat: np.zeros((total_val_size, 1, trans_mat_size)),
                is_training: False,
                z_e_p: np.zeros((total_val_size, 1, latent_dim)),
                z_p: np.concatenate([next_z_e_o, next_cond_o], axis=-1),
                init_1: np.zeros((2, batch_size, lstm_dim)),
                c: batch_conditions,
                concat_x_c: batch_concat_x_c,
            }
            final_x = sess.run(reconstruction, feed_dict=f_dic)

            x_pred_hat = np.concatenate(
                [x_pred_hat, np.reshape(final_x, (-1, 1, input_size))], axis=1
            )

    actuals = np.reshape(x_pred_hat[:, -num_pred:, :], (-1, input_size))
    predictions = np.reshape(
        data_val_full[:total_val_size, -num_pred:, trans_mat_size:],
        (-1, input_size),
    )

    # Per feature quantiles (shape: (10, input_size))
    actual_bins = np.quantile(actuals, q=[i / 10 for i in range(1, 10)], axis=0)
    prediction_bins = np.quantile(predictions, q=[i / 10 for i in range(1, 10)], axis=0)

    # Bucketing each feature(input_size)
    actual_bucketised = []
    prediction_bucketised = []
    for col in range(actuals.shape[1]):
        actual_bucketised.append(
            np.digitize(actuals[:, col], actual_bins[:, col]).reshape(-1, 1)
        )
        prediction_bucketised.append(
            np.digitize(predictions[:, col], prediction_bins[:, col]).reshape(-1, 1)
        )
    actual_bucketised = np.concatenate(actual_bucketised, axis=1)
    prediction_bucketised = np.concatenate(prediction_bucketised, axis=1)

    featurewise_deviation = np.mean(
        np.abs(actual_bucketised - prediction_bucketised), axis=0
    ).tolist()

    mse = sklearn.metrics.mean_squared_error(actuals, predictions)
    mae = sklearn.metrics.mean_absolute_error(actuals, predictions)

    metrics_dict = {
        "experiment_name": ex_name,
        "MSE_reconstruction": mse,
        "MAE_reconstruction": mae,
        "Featurewise_deviation": featurewise_deviation,
    }

    print(metrics_dict)

    return metrics_dict


@ex.capture
def evaluate_nmi(
    model,
    label_index_for_nmi,
    x,
    x_trans_mat,
    y_trans_mat,
    c_dim,
    val_gen,
    len_data_val,
    modelpath,
    epochs,
    trans_mat_size,
    c,
    concat_x_c,
    max_n_step,
    batch_size,
    som_dim,
    learning_rate,
    alpha,
    gamma,
    beta,
    theta,
    tau,
    epochs_pretrain,
    ex_name,
    kappa,
    dropout,
    prior,
    latent_dim,
    eta,
    lstm_dim,
):
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0)
    num_batches = len_data_val // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)

        is_training = model.is_training
        graph = tf.get_default_graph()
        init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
        z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
        z_e_rec = graph.get_tensor_by_name("reconstruction_e/decoder/z_e:0")
        training_dic = {
            is_training: True,
            z_e_p: np.zeros((max_n_step, batch_size, latent_dim)),
            init_1: np.zeros((2, batch_size, lstm_dim)),
            z_e_rec: np.zeros((max_n_step * batch_size, latent_dim + c_dim)),
        }

        test_k_all = []
        labels_val_all = []
        z_q_all = []
        z_e_all = []
        print("Evaluation...")

        for i in range(num_batches):
            batch_val_data, batch_val_labels, ii = next(val_gen)

            # Added for CVAE
            batch_val_ts_data = batch_val_data[:, :, trans_mat_size:]
            batch_val_trans_mat_x = batch_val_data[:, :, :trans_mat_size]
            batch_val_trans_mat_y = batch_val_labels[:, :, :trans_mat_size]
            batch_conditions = np.zeros((batch_val_data.shape[0], max_n_step, c_dim))
            batch_concat_x_c = np.concatenate(
                (batch_val_ts_data, batch_conditions), axis=2
            )

            f_dic = {
                x: batch_val_ts_data,
                x_trans_mat: batch_val_trans_mat_x,
                y_trans_mat: batch_val_trans_mat_y,
                c: batch_conditions,
                concat_x_c: batch_concat_x_c,
            }
            f_dic.update(training_dic)
            test_k_all.extend(sess.run(model.k, feed_dict=f_dic))
            labels_val_all.extend(batch_val_labels[:, :, trans_mat_size:])
            z_q_all.extend(sess.run(model.z_q, feed_dict=f_dic))
            z_e_all.extend(sess.run(model.z_e_sample, feed_dict=f_dic))

        labels_val_all = np.array(labels_val_all)
        test_k_all = np.array(test_k_all)
        labels_val_all = np.reshape(labels_val_all, (-1, labels_val_all.shape[-1]))

        print("Test k output shape:", test_k_all.shape)
        print("Test k actual shape:", labels_val_all.shape)

        print(
            "Mean: {:.3f}, Std: {:.3f}".format(
                np.mean(labels_val_all[:, label_index_for_nmi]),
                np.std(labels_val_all[:, label_index_for_nmi]),
            )
        )
        NMI_index = metrics.normalized_mutual_info_score(
            labels_val_all[:, label_index_for_nmi],
            test_k_all,
            average_method="geometric",
        )

    results = {}
    results["NMI"] = NMI_index

    f = open("results_players_counters.txt", "a+")
    f.write(
        "Epochs= %d, som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, "
        "theta= %f, eta= %f, beta= %f, alpha=%f, gamma=%f, tau=%f, epochs_pretrain=%d, dropout= %f, prior= %f"
        % (
            epochs,
            som_dim[0],
            som_dim[1],
            latent_dim,
            batch_size,
            learning_rate,
            theta,
            eta,
            beta,
            alpha,
            gamma,
            tau,
            epochs_pretrain,
            dropout,
            prior,
        )
    )
    f.write(", kappa= %f, NMI: %f. Name: %r \n" % (kappa, results["NMI"], ex_name))
    f.close()

    return results


@ex.automain
def main(
    input_size,
    max_n_step,
    latent_dim,
    som_dim,
    learning_rate,
    decay_factor,
    alpha,
    beta,
    gamma,
    theta,
    kappa,
    tau,
    prior,
    conditional_loss_weight,
    dropout,
    eta,
    train_ratio,
    lstm_dim,
    trans_mat_size,
    c_dim,
    train_val_split_ratio,
    filename,
):
    tf.reset_default_graph()

    lr_val = tf.placeholder_with_default(learning_rate, [])
    prior_val = tf.placeholder_with_default(prior, [])

    model = Mirage(
        input_size=input_size,
        latent_dim=latent_dim,
        som_dim=som_dim,
        learning_rate=lr_val,
        decay_factor=decay_factor,
        dropout=dropout,
        input_channels=input_size,
        alpha=alpha,
        beta=beta,
        eta=eta,
        kappa=kappa,
        tau=tau,
        theta=theta,
        gamma=gamma,
        prior=prior,
        conditional_loss_weight=conditional_loss_weight,
        lstm_dim=lstm_dim,
        condition_size=c_dim,
        trans_mat_size=trans_mat_size,
    )

    data_train, data_val, endpoints_total_train, endpoints_total_val = get_data(
        train_val_split_ratio, filename
    )
    (
        data_train_full,
        data_val_full,
        endpoints_total_train_full,
        endpoints_total_val_full,
    ) = get_data(train_val_split_ratio, filename)

    data_train = data_train[:, :max_n_step, :]
    data_val = data_val[:, :max_n_step, :]
    endpoints_total_train = endpoints_total_train[:, :max_n_step, :]
    endpoints_total_val = endpoints_total_val[:, :max_n_step, :]

    if train_ratio < 1.0:
        data_train = data_train[: int(len(data_train) * train_ratio)]

    results = train_model(
        model,
        data_train,
        data_val,
        endpoints_total_train,
        endpoints_total_val,
        data_train_full,
        endpoints_total_train_full,
        data_val_full,
        endpoints_total_val_full,
        lr_val,
        prior_val,
    )

    return results
