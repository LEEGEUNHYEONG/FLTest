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
        '''
               https://github.com/roxanneluo/Federated-Learning/blob/master/mnist_cnn.py
        '''

        with K.tf.device('/gpu:0'):
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

        '''
        #   Server layer 갯수 변경
        with K.tf.device('/gpu:0'):
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ])
        '''
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                               loss=keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy'])


    def set(self, train_image, train_label, weights=None, epoch=5, batch_size=10):
        with K.tf.device('/gpu:0'):
            self.set_local_weight(weights)
            train_image = train_image.reshape((-1, 28, 28, 1))
            hist = self.model.fit(train_image, train_label, epochs=epoch, batch_size=batch_size, verbose=1)
        return hist, self.model.get_weights()


    def get_weight(self):
        return self.model.get_weights()

    def set_global_weight(self, weight):
        self.model.set_weights(weight)

    def set_local_weight(self, weight_list):
        if weight_list is not None:  # todo : initialize weight  체크 방법
            self.model.set_weights(weight_list)


    def local_evaluate(self, test_image, test_label):
        test_image = test_image.reshape((-1, 28, 28, 1))
        result = self.model.predict(test_image, test_label, verbose=0)
        return result


    '''
       model.fit(train_images, train_labels, epochs=1)
       model.evaluate(test_images, test_labels)
       predictions = model.predict(t_test_images) 
    '''
