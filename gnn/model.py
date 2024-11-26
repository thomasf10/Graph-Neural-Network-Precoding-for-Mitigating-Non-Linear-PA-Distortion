import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utils.utils import rayleigh_channel_MU
from gnn.activations import get_activation
from gnn.naming import get_name

class Efficient_GNN_layer(layers.Layer):
    def __init__(self, input_feature_size, feature_size, M, K, nr=0, act=None, aggregation='sum', **kwargs):

        super(Efficient_GNN_layer, self).__init__(**kwargs)
        self.M = M
        self.K = K
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.nr = nr
        self.activation_string = act
        self.activation = get_activation(self.activation_string)
        self.aggregation = aggregation

        self.edge_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wedge_{self.nr}'
        )

        self.remainder_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wremainder_{self.nr}'
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_feature_size': self.input_feature_size,
            'feature_size': self.feature_size,
            'M': self.M,
            'K': self.K,
            'act': self.activation_string,
            'nr': self.nr,
            'aggregation': self.aggregation
        })
        return config

    def call(self, inputs):
        """
        :param inputs: bs x MK x input_feature_size
        :return: outputs: bs x MK x feature_size
        """
        batch_size = tf.shape(inputs)[0]

        # update edge features: bs x feature_size x input_feature_size @ bs x input_feature_size x MK
        Z_l_edge = self.edge_weights @ tf.transpose(inputs, perm=[0, 2, 1])

        #transpose bs x featur_size x MK to bs x MK x feature_size
        Z_l_edge = tf.transpose(Z_l_edge, perm=[0, 2, 1])

        #aggregation bs x MK x inputfeaturesize to bs x 1 x inputfeatresize
        aggregated_channel = tf.reduce_mean(inputs, axis=1, keepdims=True)
        #todo maybe remove h_m,k from this mean => different mean for each element in zl

        #multiply with a learned weight matrix
        Whagg = self.remainder_weights @ tf.transpose(aggregated_channel, perm=[0, 2, 1]) #bs x feature_size x 1
        Whagg = tf.transpose(Whagg, perm=[0, 2, 1]) #bs x 1 x feature_size

        #sum: W_edge H + Wremainder h_aggregate (broadcasting should take care of it)
        z_l = Z_l_edge + Whagg

        #activation
        z_l = self.activation(z_l)

        return z_l #, self.edge_weights, self.k_weights, self.m_weights #extra outputs for debugging

class GNN_layer(layers.Layer):
    def __init__(self, input_feature_size, feature_size, M, K, nr=0, act=None, aggregation='sum', **kwargs):

        super(GNN_layer, self).__init__(**kwargs)
        self.M = M
        self.K = K
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.nr = nr
        self.activation_string = act
        self.activation = get_activation(self.activation_string)
        self.aggregation = aggregation

        self.edge_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wedge_{self.nr}'
        )

        self.m_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wm_{self.nr}'
        )

        self.k_weights = self.add_weight(
            shape=(self.feature_size, self.input_feature_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name=f'Wk_{self.nr}'
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_feature_size': self.input_feature_size,
            'feature_size': self.feature_size,
            'M': self.M,
            'K': self.K,
            'act': self.activation_string,
            'nr': self.nr,
            'aggregation': self.aggregation
            #'name': self.layer_name
        })
        return config

    def call(self, inputs):
        """
        :param inputs: bs x MK x input_feature_size
        :return: outputs: bs x MK x feature_size
        """
        #todo maybe switch reduce_sum aggregation to reduce_mean aggregation?

        batch_size = tf.shape(inputs)[0]

        # update edge features: bs x feature_size x input_feature_size @ bs x input_feature_size x MK
        Z_l_edge = self.edge_weights @ tf.transpose(inputs, perm=[0, 2, 1])

        #transpose bs x featur_size x MK to bs x MK x feature_size
        Z_l_edge = tf.transpose(Z_l_edge, perm=[0, 2, 1])

        #aggregation
        #bs x MK x input_feature_size to bs x M x K x input_feature_size
        edges = tf.reshape(inputs, (batch_size, self.M, self.K, self.input_feature_size))

        if self.aggregation == 'sum':
            #aggregate to the antenna nodes aka for each antenna sum the edges connected to it
            m_message = tf.reduce_sum(edges, axis=2) #sum accross K dimension => bs x M x input_feature_size

            #aggregate to the user nodes aka for each user sum the edges connected to it
            k_message = tf.reduce_sum(edges, axis=1) #sum accross the M dimension => bs x K x input_feature_size
        elif self.aggregation == 'mean':
            # aggregate to the antenna nodes aka for each antenna sum the edges connected to it
            m_message = tf.reduce_mean(edges, axis=2)  # sum accross K dimension => bs x M x input_feature_size

            # aggregate to the user nodes aka for each user sum the edges connected to it
            k_message = tf.reduce_mean(edges, axis=1)  # sum accross the M dimension => bs x K x input_feature_size
        else:
            tf.Assert(False, [f'invalid aggregation operation: {self.aggregation}'])

        #multiply with a learned weight matrix
        Wm_m_message = self.m_weights @ tf.transpose(m_message, perm=[0, 2, 1]) #bs x feature_size x M
        Wk_k_message = self.k_weights @ tf.transpose(k_message, perm=[0, 2, 1]) #bs x feature_size x K
        Wm_m_message = tf.transpose(Wm_m_message, perm=[0, 2, 1]) #bs x M x feature_size
        Wk_k_message = tf.transpose(Wk_k_message, perm=[0, 2, 1]) #bs x K x feature size

        #resturcture m_message as
        # m_message[:, 0, :], ... ,  m_message[:, 0, :], m_message[:, 1, :] , ... m_message[:, M-1, :]
        m_message_expanded = tf.repeat(Wm_m_message, repeats=self.K, axis=1) #bs x KM x feature_size

        #resturcture k_message as
        # k_message[:, 0, :], k_message[:, 1, :], ..., k_message[:, K-1, :] ,k_message[:, 0, :], ... k_message[:, K-1, :]
        k_message_expanded = tf.tile(Wk_k_message, multiples=[1, self.M, 1]) #bs x KM x feature_size

        #sum all three parts
        z_l = Z_l_edge + m_message_expanded + k_message_expanded

        #activation
        z_l = self.activation(z_l)

        return z_l #, self.edge_weights, self.k_weights, self.m_weights #extra outputs for debugging

class Pwr_norm_gnn(layers.Layer):
    def __init__(self, Pt, M, K, **kwargs):
        super(Pwr_norm_gnn, self).__init__()
        self.Pt = Pt
        self.M = M
        self.K = K

    def get_config(self):
        config = super().get_config()
        config.update({
            'Pt': self.Pt,
            'M': self.M,
            'K': self.K
        })
        return config

    def call(self, inputs):
        """
        :param input: bs x MK x 2 (Real)
        :return: bs x M x K x 2 (Real)
        """
        batch_size = tf.shape(inputs)[0]

        #reshape to bs x M x K x 2
        input_reshaped = tf.reshape(inputs, (batch_size, self.M, self.K, 2))

        #cast back to complex numbers
        Wre = input_reshaped[:, :, :, 0]
        Wim = input_reshaped[:, :, :, 1]
        W = tf.complex(Wre, Wim)  # bs x K x M
        #print(f"input power: {tf.math.real(tf.norm(W, ord='fro', axis=(1, 2)) ** 2)}")

        #compute alpha
        # Wh = tf.transpose(W, perm=(0, 2, 1), conjugate=True)
        # WhW = tf.matmul(Wh, W)
        alpha = tf.math.sqrt(self.Pt / tf.math.real(tf.norm(W, ord='fro', axis=(1, 2)) ** 2))
        #alpha = tf.math.sqrt(tf.cast(self.Pt, dtype=tf.float32)) / tf.math.sqrt(tf.math.real(tf.linalg.trace(WhW)))
        alpha = tf.reshape(alpha, (tf.shape(alpha)[0], 1, 1, 1))
        #print(f'alpha: {alpha}')

        #scale the output with alpha
        output = alpha * input_reshaped #todo this does not broadcast properly fix it (see previous code)
        # Wreo = output[:, :, :, 0]
        # Wimo = output[:, :, :, 1]
        # Wo = tf.complex(Wreo, Wimo)
        # print(f"output power: {tf.math.real(tf.norm(Wo, ord='fro', axis=(1, 2)) ** 2)}")
        #print(f'ouput shape : {tf.shape(output)}')
        return output

