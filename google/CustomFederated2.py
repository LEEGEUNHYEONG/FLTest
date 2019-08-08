'''
    https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_2
    Implementing Federated Averaging
'''
# %%
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from six.moves import range

tf.compat.v1.enable_v2_behavior()


@tff.federated_computation
def hello_word():
    return "Hello, World!"


print(hello_word())
# %%
mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

print([(x.dtype, x.shape) for x in mnist_train])

# %%
'''   
    Preparing federated data sets
'''
NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100


def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples], dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
        return output_sequence


federate_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]

federate_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]

print(federate_train_data[5][-1]['y'])

# %%
from matplotlib import pyplot as plt

plt.imshow(federate_train_data[5][-1]['x'][-1].reshape(28, 28), cmap='gray')
plt.grid(False)
plt.show()

# %%
'''
    On combining TensorFlow and TensorFlowFederated
    Defining a loss functions 
'''
BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))
])
print(str(BATCH_TYPE))

# %%
MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))
])
print(str(MODEL_TYPE))


# %%
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(tf.one_hot(batch.y, 10) * tf.log(predicted_y), reduction_indices=[1]))


print(str(batch_loss.type_signature))

# %%
initial_model = {
    'weights': np.zeros([784, 10], dtype=np.float32),
    'bias': np.zeros([10], dtype=np.float32)
}

sample_batch = federate_train_data[5][-1]

print(batch_loss(initial_model, sample_batch))

# %%
'''
    Gradient descent on a single batch
'''


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    model_vars = tff.utils.get_variables('v', MODEL_TYPE)
    init_model = tff.utils.assign(model_vars, initial_model)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


print(str(batch_train.type_signature))

# %%
model = initial_model
losses = []
for _ in range(5):
    model = batch_train(model, sample_batch, 0.1)
    losses.append(batch_loss(model, sample_batch))

print(losses)

# %%
'''
    Gradient descent on a sequence of local data
'''
LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    return tff.sequence_reduce(all_batches, initial_model, batch_fn)


print(str(local_train.type_signature))

# %%
locally_trained_model = local_train(initial_model, 0.1, federate_train_data[5])
print(locally_trained_model)

# %%
'''
    Local evaluation
'''


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE), all_batches))


print(str(local_eval.type_signature))

# %%
print('initial_model_loss = ', local_eval(initial_model, federate_train_data[5]))
print('locally_trained_model_loss = ', local_eval(locally_trained_model, federate_train_data[5]))

# %%
print('initial_model_loss = ', local_eval(initial_model, federate_train_data[0]))
print('locally_trained_model_loss = ', local_eval(locally_trained_model, federate_train_data[0]))

# %%
'''
    Federated evaluation
'''
SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federate_eval(model, data):
    return tff.federated_mean(tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


print('initial_model loss = ', federate_eval(initial_model, federate_train_data))
print('locally_traind_model loss = ', federate_eval(locally_trained_model, federate_train_data))

# %%
'''
    Federated training
'''
SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)


@tff.federated_computation(SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    return tff.federated_mean(
        tff.federated_map(
            local_train, [tff.federated_broadcast(model), tff.federated_broadcast(learning_rate),
                          data]))


model = initial_model
learning_rate = 0.1
for round_num in range(5):
    model = federated_train(model, learning_rate, federate_train_data)
    learning_rate = learning_rate * 0.9
    loss = federate_eval(model, federate_train_data)

    print('round {}, loss={}'.format(round_num, loss))

# %%
print('initial_model loss = ', federate_eval(initial_model, federate_test_data))
print('trained_model loss = ', federate_eval(model, federate_test_data))
