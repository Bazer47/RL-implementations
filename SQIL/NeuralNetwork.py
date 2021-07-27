# -*- coding: utf-8 -*-
"""
Provides Neural Network class

@author: Jan Bayer
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions

class NeuralNetwork():
    ''' Contains neural network model, Q_soft and V_soft calculations
        and action sample.
    '''
    
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        # Alpha is the entropy coeff. (see Haarnoja et al. (2017))
        self.alpha = 4
        self.model = None
    
    def create_soft_q_network(self):
        ''' Creates TensorFlow model with Functional API
        '''
        # Neural network defined in Mnih et al. (Nature-2015)
        inputs = layers.Input(shape=(84, 84, 4))
    
        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    
        layer4 = layers.Flatten()(layer3)
    
        layer5 = layers.Dense(512, activation="relu")(layer4)
        layer6 = layers.Dense(self.num_actions, activation="linear")(layer5)
    
        self.model = keras.Model(inputs=inputs, outputs=layer6)
    
    def getVsoft(self, q_value):
        ''' Calculates soft value function
        '''
        v = self.alpha * tf.math.log(tf.math.reduce_sum(tf.math.exp(q_value/self.alpha),
                                                        axis=1, keepdims=True))
        return v
    
    def choose_action(self, state):
        ''' Choose action by sampling policy distribution
        '''
        state = tf.reshape(tf.convert_to_tensor(state), [1,84,84,4])
        # print('state : ', state)
        q = self.model(state, training=False)
        v = tf.squeeze(self.getVsoft(q))
        # print('q & v', q, v)
        # Generate policy distribution
        # print(q-v)
        dist = tf.math.exp((q-v)/self.alpha)
        dist = dist / tf.math.reduce_sum(dist)
        # print(dist)
        c = tfd.Categorical(dist)
        a = c.sample()
        return a