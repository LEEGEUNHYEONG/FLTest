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
def show(i):
    plt.figure(figsize=(5, 5))
    image = i
    plt.imshow(image, cmap='Greys')
    plt.show()

# %%

'''
for i in range(10) :
    print("round : {}".format(i))
    for j in range(20):
        ti, tl = make_sublist(train_index_list, train_images, train_labels, i)
        m1 = TestModel()
        if i != 0 :
            m1.set_local_weight(server.get_weight())
        m1.set(ti, tl)
        w1 = m1.get_weight()
        server.update_value(w1)
'''
#%%
'''
    mnist의 특정 숫자만 federate 학습 하도록 함 
'''
def learning_federated_number(value=0):
    ti, tl = make_sublist(train_index_list, train_images, train_labels, value)
    m1 = TestModel()
    m1.set(ti, tl, server.get_weight())
    server.update_value(m1.get_weight())


#%%
'''
    서버에서 fed_weight를 받아와 특정 숫자로 로컬 테스트 진행
'''
def evaluate_federated_number(value=-1):
    m1 = TestModel()
    m1.set(train_images, train_labels, weights=server.get_weight())
    if value == -1:
        m1.local_evaluate(test_images, test_labels)
    else:
        ti, tl = make_sublist(test_index_list, test_images, test_labels, value)
        m1.local_evaluate(ti, tl)

#%%
def evaluate_number(value = -1):
    m1 = TestModel()
    m1.set(train_images, train_labels)
    if value == -1 :
        m1.local_evaluate(test_images, test_labels)
    else:
        ti, tl = make_sublist(test_index_list, test_images, test_labels, value)
        m1.local_evaluate(ti, tl)

# %%
'''
    클라이언트의 수
    0과 9만 학습 시킴 
'''
for i in range(10):
    print("rount : {}".format(i))
    learning_federated_number(0)
    learning_federated_number(9)

#%%
evaluate_federated_number(9)

#%%
evaluate_number(9)

#%%
#server.clear_weight()

#%%
'''
m = TestModel()
ti2, tl2 = make_sublist(train_index_list, train_images, train_labels, 5)
m.set_local_weight(server.weight_list)
m.set(ti2, tl2, 1)
ti2, tl2 = make_sublist(test_index_list, test_images, test_labels, 5)
m.local_evaluate(ti2, tl2)

#%%
mainModel = TestModel()
ti2, tl2 = make_sublist(train_index_list, train_images, train_labels, 5)
#m.set_local_weight(server.weight_list)
mainModel.set(train_images, train_labels, epoch=1)
mainModel.local_evaluate(ti2, tl2)
# %%

server.clear_weight()
'''
