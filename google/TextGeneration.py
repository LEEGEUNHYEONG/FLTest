'''
https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
'''

# %%
from __future__ import absolute_import, division, print_function

import collections
import functools
import os
import time

import numpy as np
import tensorflow as tf

import tensorflow_federated as tff

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

# Test the TFF is working:
tff.federated_computation(lambda: 'Hello, World!')()

# %%
''' Generate the vocab lookup tables '''
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# %%
''' Load the pre-trained model and generate some text '''


def load_model(batch_size):
    urls = {
        1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
        8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'
    }
    assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
    url = urls[batch_size]
    local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)
    return tf.keras.models.load_model(local_file, compile=False)


# %%
def generate_text(model, start_string):
    # From https://www.tensorflow.org/tutorials/sequences/text_generation
    num_generate = 200
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# %%
keras_model_batch1 = load_model(batch_size=1)
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))

# %%
'''
    Load and Preprocess the Federated Shakespeare Data
'''
train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

raw_example_dataset = train_data.create_tf_dataset_for_client('THE_TRAGEDY_OF_KING_LEAR_KING')

for x in raw_example_dataset.take(2):
    print(x['snippets'])

# %%
SEQ_LENGTH = 100
BATCH_SIZE = 8
BUFFER_SIZE = 10000  # for data shuffling

BatchType = collections.namedtuple('BatchType', ['x', 'y'])

table = tf.contrib.lookup.index_table_from_tensor(mapping=vocab, num_oov_buckets=0, default_value=0)


def to_ids(x):
    s = tf.reshape(x['snippets'], shape=[1])
    chars = tf.string_split(s, delimiter='').values
    #ids = table.lookup(chars)  #   int type cast 문제 발생, 강제로 타입 지정해 줘야 함(?)
    ids = tf.cast(table.lookup(chars), tf.int32)
    return ids


def split_input_target(chunk):
    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    target_text = tf.map_fn(lambda x: x[1:], chunk)
    return BatchType(input_text, target_text)


def preprocess(dataset):
    return (
        dataset.map(to_ids)
            .apply(tf.data.experimental.unbatch())
            .batch(SEQ_LENGTH + 1, drop_remainder=True)
            .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
            .map(split_input_target))


# %%
example_dataset = preprocess(raw_example_dataset)
print(example_dataset.output_types, example_dataset.output_shapes)

# %%
'''
    Compile the model and test on the preprocessed data    
'''


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='accuracy', dtype=None):
        super(FlattenedCategoricalAccuracy, self).__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
        return super(FlattenedCategoricalAccuracy, self).update_state(y_true, y_pred, sample_weight)


def compile(keras_model):
    keras_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[FlattenedCategoricalAccuracy()])
    return keras_model


# %%
BATCH_SIZE = 8
keras_model = load_model(batch_size=BATCH_SIZE)

compile(keras_model)
print('Evaluating on an example Shakespeare character : ')
keras_model.evaluate(example_dataset.take(1))

random_indexes = np.random.randint(low=0, high=len(vocab), size=1 * BATCH_SIZE * (SEQ_LENGTH + 1))
data = {'snippets': tf.constant(''.join(np.array(vocab)[random_indexes]), shape=[1, 1])}
random_datasets = preprocess(tf.data.Dataset.from_tensor_slices(data))
print('Expected accuracy for random guessing: {:.3f}'.format(1.0 / len(vocab)))
print('Evaluating on completely random data : ')
keras_model.evaluate(random_datasets, steps=1)

# %%
'''
    Fine-tune the model with Federated Learning
'''
def create_tff_model():
  x = tf.constant(np.random.randint(1, len(vocab), size=[BATCH_SIZE, SEQ_LENGTH]))
  dummy_batch = collections.OrderedDict([('x', x), ('y', x)])
  keras_model_clone = compile(tf.keras.models.clone_model(keras_model))
  return tff.learning.from_compiled_keras_model(
      keras_model_clone, dummy_batch=dummy_batch)


# %%
fed_avg = tff.learning.build_federated_averaging_process(model_fn=create_tff_model)

# %%
state = fed_avg.initialize()
state, metrics = fed_avg.next(state, [example_dataset.take(1)])
print(metrics)

# %%
def data(client, source=train_data):
    return preprocess(source.create_tf_dataset_for_client(client)).take(2)

clients = ['ALL_S_WELL_THAT_ENDS_WELL_CELIA',
           'MUCH_ADO_ABOUT_NOTHING_OTHELLO',
           'THE_TRAGEDY_OF_KING_LEAR_KING']

train_datasets = [data(client) for client in clients]

test_dataset = functools.reduce(lambda d1, d2: d1.concatenate(d2),
                                [data(client, test_data) for client in clients])

# %%
NUM_ROUNDS =3
state = fed_avg.initialize()

state = tff.learning.state_with_new_model_weights(
    state,
    trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
    non_trainable_weights=[
        v.numpy() for v in keras_model.non_trainable_weights])

def keras_evaluate(state, round_num) :
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)
    print('Evaluating before training round ', round_num)
    keras_model.evaluate(example_dataset, steps=2)

for round_num in range(NUM_ROUNDS):
    keras_evaluate(state, round_num)
    state, metrics = fed_avg.next(state, train_datasets)
    print('Training metrics : ', metrics)

keras_evaluate(state, NUM_ROUNDS+1)

# %%
keras_model_batch1.set_weights([v.numpy() for v in keras_model.weights])
print(generate_text(keras_model_batch1, 'What of TensorFlow Federated, you ask? '))

