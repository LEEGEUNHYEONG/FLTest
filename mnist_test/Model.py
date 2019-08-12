import tensorflow as tf
from tensorflow import keras


class TestModel:
    train_image = []
    train_label = []
    model = None

    number_epochs = 5

    def __init__(self):
        return

    def set(self, train_image, train_label, weights=[]):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.5),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        if len(weights) != 0:
            self.model.add_weight = weights

        self.model.fit(train_image, train_label)

        return self.model.get_weights()

    def get_weight(self):
        return self.model.get_weights()

    def evaluate(self, test_image, test_label):
        self.model.evaluate(test_image, test_label)


    '''
       model.fit(train_images, train_labels, epochs=1)
       model.evaluate(test_images, test_labels)
       predictions = model.predict(t_test_images) 
    '''
