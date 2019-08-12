# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mnist_test.Server import Server
from mnist_test.Model import TestModel

server = Server.instance()

'''
#   그래픽카드 사용 확인 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

np.random.seed(42)

# %%
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# %%    해당 이미지의 labels 별로 구분하여 리스트에 저장
print("train image :", train_images.shape)
print("train label :", train_labels.shape)
print("text image : ", test_images.shape)
print("text label : ", test_labels.shape)
print('\n')

# %%
train_index_list = [[], [], [], [], [], [], [], [], [], []]
test_index_list = [[], [], [], [], [], [], [], [], [], []]

for i, v in enumerate(train_labels):
    train_index_list[v].append(i)

for i, v in enumerate(test_labels):
    test_index_list[v].append(i)

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
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=keras.optimizers.SGD(lr=0.5),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# %%
model.fit(train_images, train_labels, epochs=1)

# %%
model.evaluate(test_images, test_labels)

# %%
predictions = model.predict(t_test_images)

# %%
w = []
w.append(model.get_weights()[0])
w.append(model.get_weights()[1])
w.append(model.get_weights()[2])
w.append(model.get_weights()[3])


#server.update_value(w)


# %%
'''
    특정 숫자의 image와 label의 리스트를 생성 함    
'''
def make_sublist(index_list, images_list, labels_list, target_index):
    target_images = []
    target_labels = []

    print(index_list[target_index])
    for i, v in enumerate(index_list[target_index]):
        target_images.append(images_list[v])
        target_labels.append(labels_list[v])

    target_images = np.array(target_images)
    target_labels = np.array(target_labels)

    return target_images, target_labels

# %%
'''
    이미지를 출력
'''
def show(i) :
    plt.figure(figsize=(5, 5))
    image = i
    plt.imshow(image, cmap='Greys')
    plt.show()

# %%
ti, tl = make_sublist(train_index_list, train_images, train_labels, 1)
tti, ttl = make_sublist(test_index_list, test_images, test_labels, 1)
m1 = TestModel()
w1 = m1.set(ti, tl)
server.update_value(w1)

# %%
ti, tl = make_sublist(train_index_list, train_images, train_labels, 9)
tti, ttl = make_sublist(test_index_list, test_images, test_labels, 9)
m9 = TestModel()
w9 = m9.set(ti, tl)
rweight = server.update_value(w9)

# %%

rmodel = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

rmodel.compile(optimizer=keras.optimizers.SGD(lr=0.5),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

rmodel.set_weights(rweight)

#%%

'''
ti, tl = make_sublist(train_index_list, train_images, train_labels, 1)
tti, ttl = make_sublist(test_index_list, test_images, test_labels, 1)

rmodel.evaluate(tti, ttl)
'''


#%%
rweight = server.update_value(w9)




