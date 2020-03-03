import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from libs.modules import NodeEmbedding
from libs.modules import Predictor


class Model(tf.keras.Model):
    def __init__(self,
                 list_props,
                 num_embed_layers,
                 embed_dim,
                 readout_dim,
                 num_embed_heads=4,
                 num_readout_heads=4,
                 embed_use_ffnn=True,
                 embed_dp_rate=0.1,
                 embed_nm_type='gn',
                 num_groups=8,
                 last_activation=None):
        super(Model, self).__init__()

        self.num_embed_layers = num_embed_layers
        self.num_props = len(list_props)

        self.first_embedding = layers.Dense(embed_dim, use_bias=False)
        self.node_embedding = [NodeEmbedding(embed_dim, readout_dim, num_embed_heads, num_readout_heads,
                                             embed_use_ffnn, embed_dp_rate, embed_nm_type, num_groups)
                               for _ in range(num_embed_layers)]

        self.predictors = []
        for i in range(self.num_props):
            self.predictors.append(Predictor(readout_dim, last_activation[i], name=list_props[i]))

    def call(self, data, training):
        if type(data) == dict:
            x = data['x']
            adj = data['a']
        else:
            x, adj = data[0], data[1]
        h = self.first_embedding(x)
        z_list = []
        for i in range(self.num_embed_layers):
            h = self.node_embedding[i](h, adj, training)
            z_list.append(h)

        # other concat method? PMAreadout, weighted sum, etc
        z = tf.concat(z_list, axis=1)

        outputs = []
        for i in range(self.num_props):
            output = tf.squeeze(self.predictors[i](z))
            outputs.append(tf.reshape(output, [-1]))
        return outputs[0], outputs[1], outputs[2], outputs[3]