# %%
import matplotlib.pyplot as plt
import numpy as np, array
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%    해당 이미지의 labels 별로 구분하여 리스트에 저장
print("train image :", train_images.shape)
print("train label :", train_labels.shape)
print("text image : ", test_images.shape)
print("text label : ", test_labels.shape)
print('\n')


index_list = [[], [], [], [], [], [], [], [], [], []]

for i, v in enumerate(train_labels):
    index_list[v].append(i)
'''
plt.figure(figsize=(5, 5))
image = train_images[4]
plt.imshow(image, cmap='Greys')
plt.show()
'''
# %%
temp_images = []
temp_labels = []

for i, v in enumerate(index_list[0]):
    temp_images.append(train_images[v])
    temp_labels.append(train_labels[v])

for i, v in enumerate(index_list[9]):
    temp_images.append(train_images[v])
    temp_labels.append(train_labels[v])

temp_images = np.array(temp_images)
temp_labels = np.array(temp_labels)

# %%
#train_images, train_labels = train_images / 255.0, train_labels / 255.0
train_images, train_labels = temp_images / 255.0, temp_labels / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="SGD",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

# %%
model.evaluate(test_images, test_labels)





