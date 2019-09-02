import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import optimizers


class BaseModel:
    model = None

    def __init__(self):
        self.init_model()

    '''
        https://www.tensorflow.org/tutorials/keras/basic_regression
    '''
    def init_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[6]),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.softmax),
        ])

        '''
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        '''

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        self.model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])

        print(self.model.summary())
        return self.model

    def fit(self, train_data, train_label, weights=None, epoch=10, batch_size=10):
        self.set_local_weight(weights)
        hist = self.model.fit(train_data, train_label, epochs=epoch, batch_size=batch_size, verbose=0)
        return hist, self.get_local_weight()

    def local_evaluate(self, test_image, test_label):
        return self.model.evaluate(test_image, test_label)

    def get_local_weight(self):
        print(self.model.get_weights())
        return self.model.get_weights()

    def set_local_weight(self, weight_list):
        if weight_list is not None:
            self.model.set_weights(weight_list)