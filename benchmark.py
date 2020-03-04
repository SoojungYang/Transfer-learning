import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from absl import app
from absl import logging

from libs.utils import get_task_options
from libs.dataset import get_csv_dataset
from model import Model, BenchmarkModel
from args import *


def benchmark(_):
    # ===============================
    #     Tensorboard directory
    # ===============================
    train_log_dir = 'log/gradient_tape/' + FLAGS.prefix + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_log_dir = 'log/gradient_tape/' + FLAGS.prefix + '/valid'
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    # ===============================
    #          Load Dataset
    # ===============================
    f_name = 'HIV'
    total_ds = get_csv_dataset(FLAGS.batch_size, f_name=f_name, s_name='smiles', l_name='HIV_active')
    train_ds = total_ds.take(30000)
    test_ds = total_ds.skip(30000)

    # ===============================
    #      Load Pre-trained Model
    # ===============================
    model, optimizer = define_model()
    model.load_weights(FLAGS.ckpt_path)
    print("loaded weights of pre-trained model")
    print(model.layers)

    # ===============================
    #        Attach Predictor
    # ===============================
    benchmark_last_activation, loss, metrics = get_task_options(FLAGS.benchmark_task_type)
    model = BenchmarkModel(model,
                           readout=FLAGS.benchmark_readout,
                           dp_rate=FLAGS.benchmark_dp_rate,
                           fine_tune_at=FLAGS.fine_tune_at,
                           last_activation=benchmark_last_activation)
    print("Stacked Encoder and prediction layer")
    print(model.layers)

    # ===============================
    #          Train Model
    # ===============================
    for epoch in range(FLAGS.num_epochs):
        train_step(model, optimizer, loss, train_ds, metrics, epoch, train_summary_writer)
        evaluation_step(model, test_ds, metrics, epoch, valid_summary_writer)

    if FLAGS.save_outputs:
        print ("Save the predictions for test dataset")
        model_name = f_name
        save_outputs(model, test_ds, metrics, model_name)

    print(model.summary())
    return


def save_outputs(model, dataset, metrics, model_name):
    label_total = np.empty([0, ])
    pred_total = np.empty([0, ])

    st = time.time()
    for batch, (data, label) in enumerate(dataset):
        pred = model(data, False)

        label_total = np.concatenate((label_total, label.numpy()), axis=0)
        pred_total = np.concatenate((pred_total, pred.numpy()), axis=0)
        for metric in metrics:
            metric(label, pred)

    et = time.time()

    print ("Test ", end='')
    for metric in metrics:
        print (metric.name + ':', metric.result().numpy(), ' ', end='')
    print ("Time:", round(et - st, 3))

    for metric in metrics:
        metric.reset_states()

    np.save('./outputs/' + model_name + '_label.npy', label_total)
    np.save('./outputs/' + model_name + '_pred.npy', pred_total)

    return


def train_step(model, optimizer, loss_fn, dataset, metrics, epoch, train_summary_writer):
    st = time.time()
    # ===============================
    #          Train Model
    # ===============================
    for (batch, (data, label)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            pred = model(data, True)
            loss = loss_fn(label, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for metric in metrics:
            metric(label, pred)
    et = time.time()
    print ("Train ", end='')

    # ===============================
    #         Log Metrics
    # ===============================
    with train_summary_writer.as_default():
        for metric in metrics:
            print (metric.name + ':', metric.result().numpy(), ' ', end='')
            tf.summary.scalar(metric.name, metric.result().numpy(), step=epoch)
            train_summary_writer.flush()
    print ("Time:", round(et - st, 3))

    for metric in metrics:
        metric.reset_states()

    return


def evaluation_step(model, dataset, metrics, epoch, valid_summary_writer, mc_dropout=False):
    st = time.time()
    # ===============================
    #           Evaluate
    # ===============================
    for (batch, (data, label)) in enumerate(dataset):
        pred = None
        if mc_dropout:
            pred = [model(data, True) for _ in range(FLAGS.mc_sampling)]
            pred = tf.reduce_mean(pred, axis=0)
        else:
            pred = model(data, False)

        for metric in metrics:
            metric(label, pred)
    et = time.time()
    print ("Test ", end='')

    # ===============================
    #         Log Metrics
    # ===============================
    with valid_summary_writer.as_default():
        for metric in metrics:
            print (metric.name + ':', metric.result().numpy(), ' ', end='')
            tf.summary.scalar(metric.name, metric.result().numpy(), step=epoch)
            valid_summary_writer.flush()

    print ("Time:", round(et - st, 3))
    for metric in metrics:
        metric.reset_states()

    return


def define_model():
    last_activation = []
    for prop in FLAGS.prop:
        if FLAGS.loss_dict[prop] == 'mse':
            last_activation.append(None)
        else:
            last_activation.append(tf.nn.sigmoid)

    step = tf.Variable(0, trainable=False)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[FLAGS.decay_steps, FLAGS.decay_steps * 2],
        values=[1.0, 0.1, 0.01],
    )
    lr = lambda: FLAGS.init_lr * schedule(step)
    coeff = FLAGS.prior_length * (1.0 - FLAGS.embed_dp_rate)
    wd = lambda: coeff * schedule(step)

    model = Model(
        list_props=FLAGS.prop,
        num_embed_layers=FLAGS.num_embed_layers,
        embed_dim=FLAGS.embed_dim,
        predictor_dim=FLAGS.predictor_dim,
        num_embed_heads=FLAGS.num_embed_heads,
        num_predictor_heads=FLAGS.num_predictor_heads,
        embed_use_ffnn=FLAGS.embed_use_ffnn,
        embed_dp_rate=FLAGS.embed_dp_rate,
        embed_nm_type=FLAGS.embed_nm_type,
        num_groups=FLAGS.num_groups,
        last_activation=last_activation
    )

    optimizer = tfa.optimizers.AdamW(
        weight_decay=wd,
        learning_rate=lr,
        beta_1=FLAGS.beta_1,
        beta_2=FLAGS.beta_2,
        epsilon=FLAGS.opt_epsilon
    )

    # logP, TPSA, MR, MW
    model.compile(optimizer=optimizer,
                  loss={'output_1': keras.losses.MeanSquaredError(),
                        'output_2': keras.losses.MeanSquaredError(),
                        'output_3': keras.losses.MeanSquaredError(),
                        'output_4': keras.losses.MeanSquaredError()},
                  metrics={'output_1': keras.metrics.MeanSquaredError(),
                           'output_2': keras.metrics.MeanSquaredError(),
                           'output_3': keras.metrics.MeanSquaredError(),
                           'output_4': keras.metrics.MeanSquaredError()},
                  loss_weights={'output_1': 2., 'output_2': 2., 'output_3': 2., 'output_4': 1.}
                  )

    return model, optimizer


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=benchmark, argv=[sys.argv[0]] + unparsed)


