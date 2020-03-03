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

from libs.modules import *
from libs.utils import *
from libs.preprocess import *
from libs.dataset import *
from libs.layers import *
from model import *
from args import *


def save_outputs(model, dataset, metrics, model_name):
    label_total = np.empty([0, ])
    pred_total = np.empty([0, ])

    st = time.time()
    for batch, (x, adj, label) in enumerate(dataset):
        pred = model(x, adj, False)

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


class BenchmarkModel(tf.keras.Model):
    def __init__(self,
                 model,
                 fine_tune_at=0,
                 last_activation=None):
        super(BenchmarkModel, self).__init__()

        self.pre_trained = model.layers[:5]
        self.readout = PMAReadout(128, 2)
        self.prediction = keras.layers.Dense(32)
        self.f_prediction = keras.layers.Dense(1)
        self.last_activation = last_activation
        self.dropout = keras.layers.Dropout(FLAGS.benchmark_dp_rate)

        for layer in self.pre_trained[:fine_tune_at]:
            layer.trainable = False

    def call(self, data, training=True):
        x, adj = data['x'], data['a']
        h = self.pre_trained[0](x)
        for i in range(1, 5):
            h = self.pre_trained[i](h, adj)
        h = self.readout(h)
        h = tf.reshape(h, [-1, 128])
        h = self.dropout(self.prediction(h), training=training)
        outputs = self.last_activation(self.f_prediction(h))
        outputs = tf.squeeze(outputs)
        return outputs



def train_step(model, optimizer, loss_fn, dataset, metrics, epoch, train_summary_writer):
    st = time.time()
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

    with valid_summary_writer.as_default():
        for metric in metrics:
            print (metric.name + ':', metric.result().numpy(), ' ', end='')
            tf.summary.scalar(metric.name, metric.result().numpy(), step=epoch)
            valid_summary_writer.flush()

    print ("Time:", round(et - st, 3))
    for metric in metrics:
        metric.reset_states()

    return


def benchmark(_):
    train_log_dir = 'log/gradient_tape/' + FLAGS.prefix + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_log_dir = 'log/gradient_tape/' + FLAGS.prefix + '/valid'
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)


    train_folds = 'cv_3_tr'
    test_fold = 'cv_3'
    train_ds = get_5fold_dataset(FLAGS.batch_size, train_folds)
    test_ds = get_5fold_dataset(FLAGS.batch_size, test_fold)

    model, optimizer = define_model()
    model.load_weights(FLAGS.ckpt_path)
    print("loaded weights of pre-trained model")
    print(model.layers)

    benchmark_last_activation, loss, metrics = get_task_options('cls')

    # Transfer pre-trained weights into Benchmark Model
    model = BenchmarkModel(model, FLAGS.fine_tune_at, tf.nn.sigmoid)
    print("Stacked Encoder and prediction layer")
    print(model.layers)

    # Fine tune model
    for epoch in range(FLAGS.num_epochs):
        train_step(model, optimizer, loss, train_ds, metrics, epoch, train_summary_writer)
        evaluation_step(model, test_ds, metrics, epoch, valid_summary_writer)

    if FLAGS.save_outputs:
        print ("Save the predictions for test dataset")
        model_name = test_fold
        save_outputs(model, test_ds, metrics, model_name)

    print(model.summary())
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


