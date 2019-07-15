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

#%%
'''
    On combining TensorFlow and TensorFlowFederated
    Defining a loss functions 
'''
BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))
])
print(str(BATCH_TYPE))

#%%
MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))
])
print(str(MODEL_TYPE))

#%%
@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(tf.one_hot(batch.y, 10) * tf.log(predicted_y), reduction_indices=[1]))

print(str(batch_loss.type_signature))

#%%
initial_model = {
    'weights' : np.zeros([784, 10], dtype=np.float32),
    'bias' : np.zeros([10], dtype=np.float32)
}

sample_batch = federate_train_data[5][-1]

print(batch_loss(initial_model, sample_batch))

#%%
'''
    Gradient descent on a single batch
'''
