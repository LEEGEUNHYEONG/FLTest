# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mnist_test.Model import TestModel
import datetime

#%%
from mnist_test.Server import Server
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
def learning_federated_split_number(value):
    s_train_image, s_train_label = make_split_train_data_by_number(value)
    model = TestModel()
    model.set(s_train_image, s_train_label, server.get_weight(), batch_size=10)
    server.update_value(model.get_weight())

#%%
'''
    서버에서 fed_weight를 받아와 특정 숫자로 로컬 테스트 진행
'''
def evaluate_federated_number(value=-1):
    m1 = TestModel()
    #m1.set(train_images, train_labels, weights=server.get_weight())
    s_image, s_label = make_split_train_data()
    m1.set(s_image, s_label, weights=server.get_weight())
    if value == -1:
        #s_test_image, s_test_label = make_split_test_data()
        #m1.local_evaluate(s_test_image, s_test_label)
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
    분류된 index image 리스트에서 (train_index_list) 특정 index에서 size갯수만큼 랜덤하게 뽑아 
    분할된 이미지 리스트를 만듦, sklearn의 mnist_test_split()과 유사한 기능  
'''
def make_split_train_data_by_number(index_number, size=600):
    random_index = np.random.randint(0, high=len(train_index_list[index_number]), size=size)

    s_train_image=[]
    s_train_label=[]
    for v in random_index:
        s_train_image.append(train_images[train_index_list[index_number][v]])
        s_train_label.append(train_labels[train_index_list[index_number][v]])
    return np.array(s_train_image), np.array(s_train_label)

# %%
def make_split_train_data(size=600):
    random_index = np.random.randint(0, high=len(train_labels), size=size)

    s_train_image = []
    s_train_label = []
    for v in random_index:
        s_train_image.append(train_images[v])
        s_train_label.append(train_labels[v])
    return np.array(s_train_image), np.array(s_train_label)

# %%
def make_split_test_data(size=100):
    random_index = np.random.randint(0, high=len(test_labels), size=size)

    s_test_image = []
    s_label_label = []
    for v in random_index:
        s_test_image.append(test_images[v])
        s_label_label.append(test_labels[v])
    return np.array(s_test_image), np.array(s_label_label)

#%%
for i in range(100):
    print("round : ", i)
    for j in range(10):
        learning_federated_split_number(j)


#%%
evaluate_federated_number()

 #%%
#evaluate_number()


#%%
server.clear_weight()

