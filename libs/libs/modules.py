import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from libs.layers import *


# TODO list
# 1. weight standardization
# 2. fix group normalization

class NodeEmbedding(layers.Layer):
    def __init__(self,
                 embed_dim,
                 readout_dim,
                 num_embed_heads,
                 num_readout_heads,
                 use_ffnn,
                 dropout_rate,
                 nm_type='gn',
                 num_groups=8):
        super(NodeEmbedding, self).__init__()

        pre_act = True
        if use_ffnn:
            pre_act = False

        self.gconv = GraphAttn(embed_dim, num_embed_heads, pre_act)
        self.use_ffnn = use_ffnn
        if use_ffnn:
            self.ffnn = feed_forward_net(out_dim)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.nm_type = nm_type

        if nm_type == 'gn':
            # TODO: Fix group norm axis (channels last setup)
            self.norm = tfa.layers.GroupNormalization(groups=num_groups, axis=-1)
        else:
            self.norm = tf.keras.layers.LayerNormalization()

        self.readout = PMAReadout(readout_dim, num_readout_heads)

    def call(self, x, adj, training):
        h = x
        h = self.gconv(h, adj)

        if self.use_ffnn:
            h = self.ffnn(h)

        h = self.dropout(h, training=training)
        h += x
        h = self.norm(h)
        h = self.readout(h)
        return h


class Predictor(layers.Layer):
    def __init__(self,
                 out_dim,
                 last_activation,
                 name='output'):
        super(Predictor, self).__init__()
        self.out_dim = out_dim
        self.dense = layers.Dense(int(out_dim//2), input_shape=[out_dim])
        self.dense2 = layers.Dense(1, input_shape=[int(out_dim//2)], activation=last_activation, name=name)

    def call(self, x):
        z = self.dense(tf.reshape(x, [-1, self.out_dim]))
        z = self.dense2(z)
        return z

