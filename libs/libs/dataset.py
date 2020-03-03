import csv
import numpy as np

import tensorflow as tf

from libs.preprocess import *


def x_to_dict(x, a):
    return {'x': x, 'a': a}


def read_csv(prop, s_name, l_name, seed, shuffle):
    rand_state = np.random.RandomState(seed)
    with open('./data/' + prop + '.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        contents = np.asarray([
            (row[s_name], row[l_name]) for row in reader if row[l_name] != ''
        ])
        if shuffle:
            rand_state.shuffle(contents)
    return contents


def get_dataset(smi,
                shuffle_buffer_size,
                batch_size):

    smi = tf.data.Dataset.from_tensor_slices(smi)
    smi = smi.shuffle(shuffle_buffer_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=calc_properties,
                                 inp=[x], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(batch_size, padded_shapes=([], [], [], []))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_single_dataset(smi,
                shuffle_buffer_size,
                batch_size):

    smi = tf.data.Dataset.from_tensor_slices(smi)
    smi = smi.shuffle(shuffle_buffer_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=logP_benchmark,
                                 inp=[x], Tout=tf.float32),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(batch_size, padded_shapes=([]))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_benchmark_dataset(smi,
                          property_func,
                          shuffle_buffer_size,
                          batch_size):

    smi = tf.data.Dataset.from_tensor_slices(smi)
    smi = smi.shuffle(shuffle_buffer_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_smiles_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=property_func,
                                 inp=[x], Tout=tf.float32),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y = y.padded_batch(batch_size, padded_shapes=([]))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_5fold_dataset(batch_size, folds, seed=123, shuffle=True):
    smi = read_csv(folds, 'smiles', 'tox', seed, shuffle)
    smi = tf.data.Dataset.from_tensor_slices(smi)

    smi = smi.prefetch(tf.data.experimental.AUTOTUNE)
    smi = smi.shuffle(buffer_size=10 * batch_size)
    ds = smi.map(
        lambda x: tf.py_function(func=convert_tox_to_graph,
                                 inp=[x],
                                 Tout=[tf.float32, tf.float32]),
        num_parallel_calls=7
    )

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 58], [None, None]))
    ds = ds.map(x_to_dict)

    y = smi.map(
        lambda x: tf.py_function(func=get_label,
                                 inp=[x],
                                 Tout=tf.float32),
        num_parallel_calls=7
    )
    y = y.padded_batch(batch_size, padded_shapes=([]))
    ds = tf.data.Dataset.zip((ds, y))
    ds = ds.cache()
    return ds
