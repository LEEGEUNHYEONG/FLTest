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
            tf.keras.layers.Dense(64, input_dim=6, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        self.model.compile(loss=tf.keras.losses.mean_squared_error,
                           optimizer=optimizer,
                           #optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                           metrics=['mae', 'mse']
                           )

        print(self.model.summary())
        return self.model

    def fit(self, train_data, train_label, weights=None, epoch=10, batch_size=10):
        #self.set_local_weight(weights)
        print(train_data, train_label)
        history = self.model.fit(train_data, train_label, epochs=epoch, batch_size=batch_size, verbose=1,
                                 validation_split=0.2)
        return history

    def local_evaluate(self, test_image, test_label):
        return self.model.evaluate(test_image, test_label)

    def get_local_weight(self):
        print(self.model.get_weights())
        return self.model.get_weights()

    def set_local_weight(self, weight_list):
        print("??? : ", weight_list)
        if weight_list is not None:
            print("update local weight")
            self.model.set_weights(weight_list)
        else:
            print("!!!!!!!!!!!!")