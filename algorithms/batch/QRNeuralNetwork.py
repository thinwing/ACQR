import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
# import tensorflow_addons as tfa

import sys
sys.path.append('.../')
from configuration.config import *

def pinball_loss_alpha(alpha:float):
    # alpha \in (0, 1)
    def pinball_loss(y_true, y_pred):
        error = y_true - y_pred

        one = tf.ones(tf.shape(error))

        error_temp_1 = tf.where(error > 0, alpha * one, one)
        error_temp_2 = tf.where(error < 0, (alpha - 1) * one, one)

        loss = error * error_temp_1 * error_temp_2

        return tf.reduce_mean(loss)

    return pinball_loss

class QRNN():
    def __init__(self, alpha, input_train, output_train):
        self.input_train = input_train
        self.output_train = output_train
        self.input_dim = len(input_train[0])
        self.output_dim = len(output_train[0])
        self.num_hidden_layer = num_hidden_layer
        self.lr = lr
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.validation = validation
        self.alpha = alpha

        # self.activation = tf.nn.relu()
        self.loss0 = pinball_loss_alpha(alpha=alpha[0])
        self.loss1 = pinball_loss_alpha(alpha=alpha[1])

    def pre_learning(self):
        ## Lower case
        # model
        self.optimizer = adam_v2(self.lr)

        self.model0 = Sequential()
        self.model0.add(Dense(self.num_hidden_layer, input_dim=self.input_dim, activation=self.activation))
        self.model0.add(Dropout(self.dropout))
        self.model0.add(Dense(self.num_hidden_layer, activation=self.activation))
        self.model0.add(Dropout(self.dropout))
        self.model0.add(Dense(self.output_dim))
        self.model0.compile(loss=self.loss0, optimizer=self.optimizer)

        # Fitting
        self.model0.fit(
                self.input_train, self.output_train,
                batch_size = None,
                epochs = self.epochs,
                validation_split = self.validation,
        )

        ## Higher case
        # model
        self.optimizer = adam_v2(self.lr)

        self.model1 = Sequential()
        self.model1.add(Dense(self.num_hidden_layer, input_dim=self.input_dim, activation=self.activation))
        self.model1.add(Dropout(self.dropout))
        self.model1.add(Dense(self.num_hidden_layer, activation=self.activation))
        self.model1.add(Dropout(self.dropout))
        self.model1.add(Dense(self.output_dim))
        self.model1.compile(loss=self.loss1, optimizer=self.optimizer)

        # Fitting
        self.model1.fit(
                self.input_train, self.output_train,
                batch_size = None,
                epochs = self.epochs,
                validation_split = self.validation,
        )

    def predict(self, input_test):
        lower = self.model0.predict(input_test)
        higher = self.model1.predict(input_test)

        result = np.vstack((lower.T, higher.T))

        return result