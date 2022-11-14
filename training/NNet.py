import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import logging
import coloredlogs
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *

CRAZYHOUSE_NNET = 13

args = {
    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 7,
    'batch_size': 150,
    'cuda': True,
    'num_channels': 256,
    'num_residual_layers': CRAZYHOUSE_NNET
}

def value_loss_fn(y_true, y_pred):
    return 0.01 * tf.square(y_true - y_pred)

def policy_loss_fn(y_true, y_pred):
    return -tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred)), 1)

class NNet():
    def __init__(self):
        # Neural Net
        # Inputs
        self.input_boards = Input(shape=(34,64))
        inputs = Reshape((34, 8, 8))(self.input_boards)

        conv0 = Conv2D(args['num_channels'], kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
        bn0 = BatchNormalization()(conv0)
        t = Activation('relu')(bn0)

        for i in range(args['num_residual_layers']):
            convX = Conv2D(128 + 64 * i, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(t)
            bnX = BatchNormalization()(convX)
            reluX = Activation('relu')(bnX)

            convX = DepthwiseConv2D(kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(reluX)
            bnX = BatchNormalization()(convX)
            reluX = Activation('relu')(bnX)

            convX = Conv2D(256, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(reluX)
            bnX = BatchNormalization()(convX)

            t = Add()([bnX, t])

        # value head
        value_head = Conv2D(8, kernel_size=1, strides=1, padding="same", data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(t)
        value_head = BatchNormalization()(value_head)
        value_head = Activation('relu')(value_head)

        value_head = Flatten()(value_head)
        value_head = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(value_head)
        value_head = Activation('relu')(value_head)

        value_head = Dense(1, activation='tanh', name='v')(value_head)

        # policy head
        policy_head = Conv2D(256, kernel_size=3, strides=1, padding='same', data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(t)
        policy_head = BatchNormalization()(policy_head)
        policy_head = Activation('relu')(policy_head)

        policy_head = Conv2D(81, kernel_size=3, strides=1, padding='same', data_format="channels_first", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = Dense(5184, activation='softmax', name='pi')(policy_head)


        self.pi = policy_head
        self.v = value_head

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=[policy_loss_fn, value_loss_fn], optimizer=SGD(learning_rate=0.00001, momentum=0.95, nesterov=True))

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args['batch_size'], epochs = args['epochs'])

    def predict(self, board):
        # preparing input
        input_rep = board.inputRepresentation()
        input_rep = input_rep[np.newaxis, :, :]

        # run
        pi, v = self.model.predict(input_rep)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        self.model.load_weights(filepath)
        log.info('Loading Weights...')
