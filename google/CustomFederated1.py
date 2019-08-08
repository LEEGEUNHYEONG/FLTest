'''
    https://www.tensorflow.org/federated/tutorials/custom_federated_algorithms_1
'''
# %%
from __future__ import absolute_import, division, print_function

import collections

import tensorflow as tf
import tensorflow_federated as tff

tf.enable_resource_variables()


@tff.federated_computation
def hello_word():
    return "Hello, World!"


print(hello_word())

# %%
federated_float_on_clients = tff.FederatedType(tf.float32, tff.CLIENTS)
print(str(federated_float_on_clients.member))
print(str(federated_float_on_clients.placement))
print(str(federated_float_on_clients))
print(federated_float_on_clients.all_equal)
print(tff.FederatedType(tf.float32, tff.CLIENTS, all_equal=True))

# %%
simple_regression_model_type = (
    tff.NamedTupleType([('a', tf.float32), ('b', tf.float32)])
)
print(str(simple_regression_model_type))
print(str(tff.FederatedType(simple_regression_model_type, tff.CLIENTS, all_equal=True)))


# %%
@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def get_average_temperature(sensor_readings):
    # print('Getting traced, the argument is "{}".'.format(type(sensor_readings).__name__))

    return tff.federated_mean(sensor_readings)


print(str(get_average_temperature.type_signature))

# %%
print(get_average_temperature([68.5, 70.3, 69.8]))


# %%
@tff.tf_computation(tf.float32)
def add_half(x):
    return tf.add(x, 0.5)


print(str(add_half.type_signature))


# %%
@tff.federated_computation(tff.FederatedType(tf.float32, tff.CLIENTS))
def add_half_on_clients(x):
    return tff.federated_map(add_half, x)


print(str(add_half_on_clients.type_signature))
# %%
add_half_on_clients([1.0, 3.0, 2.0])

# %%
try:
    def get_constant_10():
        return tf.constant(10.)


    @tff.tf_computation(tf.float32)
    def add_ten(x):
        return x + get_constant_10()

except Exception as err:
    print(err)

print(add_ten(5.0))

# %%
float32_sequence = tff.SequenceType(tf.float32)
print(str(float32_sequence))


# %%
@tff.tf_computation(tff.SequenceType(tf.float32))
def get_local_temperature_average(local_temperatures):
    sum_and_count = (
        local_temperatures.reduce((0.0, 0), lambda x, y: (x[0] + y, x[1] + 1)))
    return sum_and_count[0] / tf.cast(sum_and_count[1], tf.float32)


print(str(get_local_temperature_average.type_signature))
print(get_local_temperature_average([68.5, 70.3, 69.8]))

# %%
import numpy as np


@tff.tf_computation(tff.SequenceType(tf.int32))
def foo(x):
    return x.reduce(np.int32(0), lambda x, y: x + y)


print(foo([1, 2, 3]))


# %%
@tff.tf_computation(tff.SequenceType(collections.OrderedDict([('A', tf.int32), ('B', tf.int32)])))
def foo(ds):
    print('output_types = {}, shapes = {}'.format(
        tf.compat.v1.data.get_output_types(ds),
        tf.compat.v1.data.get_output_shapes(ds)))

    return ds.reduce(np.int32(0), lambda total, x: total + x['A'] * x['B'])


output_type = collections.OrderedDict()
print(str(foo.type_signature))

print(foo([{'A': 2, 'B': 3}, {'A': 4, 'B': 5}]))

#%%
#   Putting it all together
@tff.federated_computation(tff.FederatedType(tff.SequenceType(tf.float32), tff.CLIENTS))
def get_global_temperature_average(sensor_readings):
    return tff.federated_mean(tff.federated_map(get_local_temperature_average, sensor_readings))

print(str(get_global_temperature_average.type_signature))

print(get_global_temperature_average([[68.0, 70.0], [71.0], [68.0, 72.0, 70.0]]))
