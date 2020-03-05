import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from libs.modules import NodeEmbedding
from libs.modules import Predictor


class Model(tf.keras.Model):
    def __init__(self,
                 list_props,
                 gconv_type='GAT',
                 predictor_readout='pma',
                 num_embed_layers=4,
                 embed_dim=64,
                 predictor_dim=256,
                 num_embed_heads=4,
                 num_predictor_heads=4,
                 embed_use_ffnn=True,
                 embed_dp_rate=0.1,
                 embed_nm_type='gn',
                 num_groups=8,
                 last_activation=None):
        """
        :param list_props: list of properties, ['logP', 'TPSA', 'MR', 'MW']
        :param gconv_type: for node embedding. GAT or GCN
        :param predictor_readout: readout method. pma or sum
        :param num_embed_layers: number of gconv layers
        :param embed_dim: dim for node embedding
        :param predictor_dim: dim for readout vector
        :param num_embed_heads: num head for gconv attention
        :param num_predictor_heads: num head for readout attention
        :param embed_use_ffnn: use feedforward neural network?
        :param embed_dp_rate: dropout rate for node embedding
        :param embed_nm_type: normalization type, gn, ln or None
        :param num_groups: num groups for gn
        :param last_activation: last activation 
        """
        super(Model, self).__init__()

        self.num_embed_layers = num_embed_layers
        self.num_props = len(list_props)

        self.first_embedding = layers.Dense(embed_dim, use_bias=False)
        self.node_embedding = [NodeEmbedding(embed_dim, num_embed_heads,
                                             embed_use_ffnn, embed_dp_rate, embed_nm_type, num_groups, gconv_type)
                               for _ in range(num_embed_layers)]

        self.predictors = []
        for i in range(self.num_props):
            self.predictors.append(Predictor(predictor_readout, num_predictor_heads, predictor_dim, last_activation[i], name=list_props[i]))

    def call(self, data, training):
        if type(data) == dict:
            x = data['x']
            adj = data['a']
        else:
            x, adj = data[0], data[1]
        h = self.first_embedding(x)
        for i in range(self.num_embed_layers):
            h = self.node_embedding[i](h, adj, training)
        outputs = []
        for i in range(self.num_props):
            output = tf.squeeze(self.predictors[i](h))
            outputs.append(tf.reshape(output, [-1]))
        return outputs[0], outputs[1], outputs[2], outputs[3]


class BenchmarkModel(tf.keras.Model):
    def __init__(self,
                 model,
                 readout,
                 dp_rate,
                 fine_tune_at=0,
                 last_activation=None):
        super(BenchmarkModel, self).__init__()

        self.pre_trained = model.layers[:5]
        if readout == 'pma':
            self.readout = PMAReadout(128, 2)
        else:
            self.readout = LinearReadout(128, 'sum')
        self.prediction1 = keras.layers.Dense(16, activation=tf.keras.activations.elu)
        self.prediction2 = keras.layers.Dense(1, activation=last_activation)
        self.dropout = keras.layers.Dropout(dp_rate)

        for layer in self.pre_trained[:fine_tune_at]:
            layer.trainable = False

    def call(self, data, training=True):
        x, adj = data['x'], data['a']

        # FIRST EMBEDDING
        h = self.pre_trained[0](x)

        # NODE EMBEDDING
        for i in range(1, 5):
            h = self.pre_trained[i](h, adj)

        # READOUT
        h = self.readout(h)
        h = tf.reshape(h, [-1, 128])

        # DENSE LAYERS
        h = self.prediction1(h)
        h = self.dropout(h, training=training)
        outputs = self.prediction2(h)
        outputs = tf.squeeze(outputs)
        return outputs