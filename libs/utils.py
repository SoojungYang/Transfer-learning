import os
import re

import tensorflow as tf
import tensorflow_addons as tfa

from libs.lr_scheduler import WarmUpSchedule
from libs.layers import ws_reg


def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(4):
        os.system('nvidia-smi -i ' + str(i) + ' | grep "No running" | wc -l > empty_gpu_check')
        f = open('empty_gpu_check')
        out = int(f.read())
        if int(out) == 1:
            empty.append(i)
    if len(empty) < ngpus:
        print('avaliable gpus are less than required')
    cmd = ''
    for i in range(ngpus):
        cmd += str(empty[i]) + ','
    return cmd


def get_learning_rate_scheduler(lr_schedule='stair',
                                graph_dim=256,
                                warmup_steps=1000,
                                init_lr=1e-3,
                                decay_steps=500,
                                decay_rate=0.1,
                                staircase=True):
    scheduler = None
    if lr_schedule == 'warmup':
        scheduler = WarmUpSchedule(
            d_model=graph_dim,
            warmup_steps=warmup_steps
        )

    elif lr_schedule == 'stair':

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )

    return scheduler


def focal_loss_fn():
    alpha = float(loss_type.split('_')[1])
    gamma = float(loss_type.split('_')[2])
    loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
        from_logits=False, alpha=alpha, gamma=gamma
    )
    return loss_fn


def get_task_options(benchmark_task_type):
    """
    :param benchmark_task_type:
    :return:  last_activation, loss, metrics
    """
    if benchmark_task_type == 'reg':
        return None, tf.keras.losses.MeanSquaredError(), [
            tf.keras.metrics.MeanAbsoluteError(name='MAE'),
            tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
        ]
    else:
        return tf.nn.sigmoid, tf.keras.losses.BinaryCrossentropy(), [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            tf.keras.metrics.AUC(curve='ROC', name='AUROC'),
            tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall'),
        ]


def get_regularizer(reg_type, wd=0.0):
    if reg_type == 'ws_reg':
        return ws_reg
    elif reg_type == 'l2_reg':
        return tf.keras.regularizers.l2(l=wd)
    else:
        return None
