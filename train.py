import argparse
import os
import sys
import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from absl import app
from absl import logging

from libs.utils import set_cuda_visible_device, get_regularizer
from libs.dataset import get_multitask_dataset
from model import Model
from args import *

FLAGS = None
np.set_printoptions(3)
tf.random.set_seed(1234)

cmd = set_cuda_visible_device(1)
print("Using ", cmd[:-1], "-th GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = cmd[:-1]


def train(model, smi):
    model_name = FLAGS.prefix
    model_name += '_' + str(FLAGS.num_embed_layers)
    model_name += '_' + str(FLAGS.embed_dim)
    model_name += '_' + str(FLAGS.predictor_dim)
    model_name += '_' + str(FLAGS.num_embed_heads)
    model_name += '_' + str(FLAGS.num_predictor_heads)
    model_name += '_' + str(FLAGS.embed_dp_rate)
    ckpt_path = './save/' + model_name + '.ckpt'
    tsbd_path = './log/' + model_name

    # ===============================
    #         Load Dataset
    # ===============================
    num_train = int(len(smi) * 0.8)
    test_smi = smi[num_train:]
    train_smi = smi[:num_train]
    train_ds = get_multitask_dataset(train_smi, FLAGS.batch_size)
    test_ds = get_multitask_dataset(test_smi, FLAGS.batch_size)

    # ===============================
    # Learning Rate and Weight Decay
    # ===============================
    step = tf.Variable(0, trainable=False)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[FLAGS.decay_steps, FLAGS.decay_steps * 2],
        values=[1.0, 0.5, 0.1],
    )
    lr = lambda: FLAGS.init_lr * schedule(step)
    coeff = FLAGS.prior_length * (1.0 - FLAGS.embed_dp_rate)
    # wd = lambda: coeff * schedule(step)

    regularizer = get_regularizer(FLAGS.reg_type, coeff)
    decay_attributes = ['kernel_regularizer', 'bias_regularizer',
                        'beta_regularizer', 'gamma_regularizer']

    for layer in model.layers:
        for attr in decay_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # ===============================
    #          Optimizer
    # ===============================
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                         beta_1=FLAGS.beta_1,
                                         beta_2=FLAGS.beta_2,
                                         epsilon=FLAGS.opt_epsilon)

    # ==========================================
    #   Compile Model with Losses and Metrics
    # ==========================================
    metric_list = [keras.metrics.MeanSquaredError(), keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    model.compile(optimizer=optimizer,
                  loss={'output_1': keras.losses.MeanSquaredError(),
                        'output_2': keras.losses.MeanSquaredError(),
                        'output_3': keras.losses.MeanSquaredError(),
                        'output_4': keras.losses.MeanSquaredError()},
                  metrics={'output_1': metric_list,
                           'output_2': metric_list,
                           'output_3': metric_list,
                           'output_4': metric_list},
                  loss_weights={'output_1': 2., 'output_2': 2., 'output_3': 2., 'output_4': 1.}
                  )

    # ===============================
    #          Callbacks
    # ===============================
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=False,
            save_freq='epoch',
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=tsbd_path,
            histogram_freq=1,
            embeddings_freq=1,
            update_freq='epoch'
        )
    ]

    st_total = time.time()
    print("model compiled and callbacks set")

    # ===============================
    #          Train Model
    # ===============================
    history = model.fit(train_ds,
                        epochs=FLAGS.num_epochs,
                        callbacks=callbacks,
                        validation_data=test_ds)
    print('\n', history.history)

    et_total = time.time()
    print("Total time for training:", round(et_total - st_total, 3))
    return


def main(_):
    def print_model_spec():
        print("Target property", FLAGS.prop)
        print("Random seed for data spliting", FLAGS.seed)
        print("Number of graph convolution layers for node embedding", FLAGS.num_embed_layers)
        print("Dimensionality of node embedding features", FLAGS.embed_dim)
        print("Dimensionality of graph features for fine-tuning", FLAGS.predictor_dim)
        print()
        print("Number of attention heads for node embedding", FLAGS.num_embed_heads)
        print("Number of attention heads for fine-tuning", FLAGS.num_predictor_heads)
        print("Type of normalization", FLAGS.embed_nm_type)
        print("Whether to use feed-forward network", FLAGS.embed_use_ffnn)
        print("Dropout rate", FLAGS.embed_dp_rate)
        print("Weight decay coeff", FLAGS.prior_length)
        print()
        return

    # ===============================
    #     Set Up Last Activation
    # ===============================
    last_activation = []
    for prop in FLAGS.prop:
        if FLAGS.loss_dict[prop] == 'mse':
            last_activation.append(None)
        else:
            last_activation.append(tf.nn.sigmoid)

    # ===============================
    #          Define Model
    # ===============================
    model = Model(
        list_props=FLAGS.prop,
        gconv_type=FLAGS.gconv_type,
        predictor_readout=FLAGS.predictor_readout,
        num_embed_layers=FLAGS.num_embed_layers,
        embed_dim=FLAGS.embed_dim,
        predictor_dim=FLAGS.predictor_dim,
        num_embed_heads=FLAGS.num_embed_heads,
        num_predictor_heads=FLAGS.num_predictor_heads,
        embed_use_ffnn=FLAGS.embed_use_ffnn,
        embed_dp_rate=FLAGS.embed_dp_rate,
        embed_nm_type=FLAGS.embed_nm_type,
        num_groups=FLAGS.num_gn_groups,
        last_activation=last_activation
    )
    print_model_spec()

    # ===============================
    #    Load Data and Train Model
    # ===============================
    smi_data = open('./data/smiles_pretrain.txt', 'r')
    smi_data = [line.strip('\n') for line in smi_data.readlines()]
    train(model, smi_data)
    return


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
