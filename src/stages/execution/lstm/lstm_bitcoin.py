import numpy
import os
import random
import tensorflow as tf
from tensorflow_core import metrics

from keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow_core import metrics
from numpy import random

from src.stages.execution.builder import Builder


class LSTMBitcoin(Builder):

    def __init__(self, config_file: str):
        super().__init__(config_file)

    def build(self):
        tf.random.set_seed(7)
        numpy.random.seed(7)
        os.environ['PYTHONHASHSEED'] = str(7)
        random.seed(7)
        self.model.add(LSTM(37, input_shape=(self.get_attributes('past_steps'), self.get_attributes('features'))))
        self.model.add(Dense(self.get_attributes('future_steps')))

        self.model.compile(loss='mae', optimizer='adam',
                           metrics=[metrics.RootMeanSquaredError(name='rmse')])
        self.model.summary()
