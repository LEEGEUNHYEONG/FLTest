# %%
from __future__ import absolute_import, division, print_function

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

tf.compat.v1.enable_v2_behavior()

tff.federated_computation(lambda: 'Hello, World!')()

# %% data load
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

print(len(emnist_test.client_ids))

emnist_train.output_types, emnist_train.output_shapes

# %%
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = iter(example_dataset).next()
example_element['label'].numpy()

# %%
from matplotlib import pyplot as plt

plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid('off')
_ = plt.show()

# %%
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500


def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1]))
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)


preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

sample_batch


# %%
def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x))
            for x in client_ids]


NUM_CLIENTS = 3

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)
len(federated_train_data), federated_train_data[0]


# %%
def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


iterative_process = tff.learning.build_federated_averaging_process(model_fn)

#   todo : 더이상 진행 불가, tensorflow-federated 버전 업데이트 됬지만 pip에서 업데이트 안됨

