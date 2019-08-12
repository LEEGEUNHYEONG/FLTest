import tensorflow as tf
from tensorflow import keras


class TestModel:
    train_image = []
    train_label = []
    model = None

    number_epochs = 10

    def __init__(self):
        return

    def set(self, train_image, train_label):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.5),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        self.model.fit(train_image, train_label, epochs=self.number_epochs)
        return self.model.get_weights()

    def evaluate(self, test_image, test_label):
        self.model.evaluate(test_image, test_label)
        return self.model.get_weights()

