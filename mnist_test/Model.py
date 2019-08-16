import keras.backend.tensorflow_backend as K
import tensorflow as tf
from tensorflow import keras


class TestModel:
    train_image = []
    train_label = []
    model = None

    def __init__(self):
        self.init_model()
        return

    def init_model(self):
        with K.tf.device('/gpu:0'):
            '''
                https://github.com/roxanneluo/Federated-Learning/blob/master/mnist_cnn.py
            '''
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),

            ])
            '''            
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)            
            '''

            self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                               loss=keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy'],
                               )

    def set(self, train_image, train_label, weights=[], epoch=5, batch_size=32):
        with K.tf.device('/gpu:0'):
            self.set_local_weight(weights)
            train_image = train_image.reshape((-1, 28, 28, 1))
            self.model.fit(train_image, train_label, epochs=epoch, batch_size=batch_size)

            return self.model

    def get_weight(self):
        return self.model.get_weights()

    def set_local_weight(self, weight_list):
        if len(weight_list) != 0:
            self.model.set_weights(weight_list)

    def local_evaluate(self, test_image, test_label):
        test_image = test_image.reshape((-1, 28, 28, 1))
        self.model.evaluate(test_image, test_label)

    '''
       model.fit(train_images, train_labels, epochs=1)
       model.evaluate(test_images, test_labels)
       predictions = model.predict(t_test_images) 
    '''
