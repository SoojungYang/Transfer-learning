import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

from libs.layers import *


class NodeEmbedding(layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_embed_heads,
                 use_ffnn,
                 dropout_rate,
                 nm_type='gn',
                 num_groups=8,
                 gconv_type='GAT'):
        super(NodeEmbedding, self).__init__()

        pre_act = True
        if use_ffnn:
            pre_act = False

        if gconv_type == 'GAT':
            self.gconv = GraphAttn(embed_dim, num_embed_heads, pre_act)
        else:
            self.gconv = GraphConv(embed_dim, pre_act)

        self.use_ffnn = use_ffnn
        if use_ffnn:
            self.ffnn = feed_forward_net(out_dim)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # ===============================
        #         Normalization
        # ===============================
        if nm_type == 'gn':
            self.norm = tfa.layers.GroupNormalization(groups=num_groups, axis=-1)
        else:
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, adj, training):
        h = x
        h = self.gconv(h, adj)

        if self.use_ffnn:
            h = self.ffnn(h)

        h = self.dropout(h, training=training)
        h += x
        h = self.norm(h)
        return h


class Predictor(layers.Layer):
    def __init__(self,
                 predictor_readout,
                 num_heads,
                 out_dim,
                 last_activation,
                 name='output'):
        super(Predictor, self).__init__()
        self.out_dim = out_dim
        if predictor_readout == 'pma':
            self.readout = PMAReadout(out_dim, num_heads)
        elif predictor_readout == 'sum':
            self.readout = LinearReadout(out_dim, 'sum')
        self.dense = layers.Dense(int(out_dim//2), input_shape=[out_dim])
        self.dense2 = layers.Dense(1, input_shape=[int(out_dim//2)], activation=last_activation, name=name)

    def call(self, x):
        x = self.readout(x)
        z = self.dense(tf.reshape(x, [-1, self.out_dim]))
        z = self.dense2(z)
        return z

