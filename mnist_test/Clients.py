# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

np.random.seed(42)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%    해당 이미지의 labels 별로 구분하여 리스트에 저장
print("train image :", train_images.shape)
print("train label :", train_labels.shape)
print("text image : ", test_images.shape)
print("text label : ", test_labels.shape)
print('\n')

train_index_list = [[], [], [], [], [], [], [], [], [], []]
test_index_list = [[], [], [], [], [], [], [], [], [], []]

for i, v in enumerate(train_labels):
    train_index_list[v].append(i)

for i, v in enumerate(test_labels):
    test_index_list[v].append(i)

plt.figure(figsize=(5, 5))
'''
plt.figure(figsize=(5, 5))
image = train_images[4]
plt.imshow(image, cmap='Greys')
plt.show()
'''
# %%
t_train_images = []
t_train_labels = []
t_test_images = []
t_test_labels = []

for i, v in enumerate(train_index_list[0]):
    t_train_images.append(train_images[v])
    t_train_labels.append(train_labels[v])

for i, v in enumerate(test_index_list[0]):
    t_test_images.append(test_images[v])
    t_test_labels.append(test_labels[v])

for i, v in enumerate(train_index_list[9]):
    t_train_images.append(train_images[v])
    t_train_labels.append(train_labels[v])

for i, v in enumerate(test_index_list[9]):
    t_test_images.append(test_images[v])
    t_test_labels.append(test_labels[v])

t_train_images = np.array(t_train_images)
t_train_labels = np.array(t_train_labels)
t_test_images = np.array(t_test_images)
t_test_labels = np.array(t_test_labels)

# %%
train_images, train_labels = train_images / 255.0, train_labels / 255.0
#train_images, train_labels = t_train_images / 255.0, t_train_labels / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5)

# %%
predictions = model.predict(test_images, test_labels)
p_index = np.argmax(predictions[0])
print(p_index, test_labels[test_labels[0]])

# %%
result = model.evaluate(test_images, test_labels)
model.lo

# %%
print(model.weights[0], model.weights[1], model.weights[2], model.weights[3])

# %%
from mnist_test.Server import Server

Server.avg(v=model.weights)
print("count : {}\navg : {}".format(Server.count, Server.avg()))
