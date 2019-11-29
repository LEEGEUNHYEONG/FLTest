# %%
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, \
    classification_report
from tensorflow import keras

import Base.BaseServer as Server

# %%
server = Server.BaseServer.instance()

#   그래픽카드 사용 확인 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


np.random.seed(42)

# %%
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255, test_images / 255

#   cnn인 경우 reshape 필요
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
input_shape = ((-1, 28, 28, 1))

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
def build_model():

    '''
    model = tf.keras.models.Sequential([
       tf.keras.layers.Flatten(input_shape=(28, 28)),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    '''



    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model


# %%
keras.utils.plot_model(build_model(), "model.png", show_shapes=True)


# %%

def train():
    model = build_model()
    result = model.fit(train_images, train_labels, batch_size=64, epochs=250)

    score = model.evaluate(test_images, test_labels, verbose=2)

    print("test acc :{}, test loss :{}".format(score[1], score[0]))
    return model


start_time = time.time()
normal_nn = train()
#normal_cnn = train()
print("time : {}".format(time.time() - start_time))

# %%
print(normal_nn.history.history['accuracy'])
test_loss, test_acc = normal_nn.evaluate(test_images, test_labels)

print("y_pred : " , test_acc)

# %%
test_loss, test_acc = normal_nn.evaluate(test_images, test_labels)

print("y_pred : " , test_acc)


# %%
print(normal_nn.history.history["accuracy"])

# %%
y_pred = np.argmax(y_pred, axis=-1)



f1 = f1_score(test_labels, y_pred, average="macro")
print("Test f1 score : %s "% f1)

acc = accuracy_score(test_labels, y_pred)
print("Test accuracy score : %s "% acc)

cm = confusion_matrix(test_labels, y_pred)
print(cm)

measure2 = precision_recall_fscore_support(test_labels, y_pred)
print("Test measure 2 : {}".format(measure2))

measure3 = classification_report(test_labels, y_pred)
print("Test measure 3 : \n{}".format(measure3))

# %%
y_pred = result.predict(test_images).argmax(axis=1)
# y_actual = np.asarray(test_labels.argmax(axis=1)).reshape(len(test_labels))
# print(metrics.classification_report(y_actual, y_pred))

# %%
print("None : ", metrics.f1_score(test_labels, y_pred, average=None))
print("micro : ", metrics.f1_score(test_labels, y_pred, average='micro'))
print("macro : ", metrics.f1_score(test_labels, y_pred, average='macro'))
print("weighted : ", metrics.f1_score(test_labels, y_pred, average='weighted'))
# print(metrics.f1_score(test_labels, y_pred, average='samples'))


# %%
round_result_acc = []
round_result_loss = []
round_result_time = []
total_time = []


def run_federated(user_number=1, round=1, batch_size=5, epochs=10):
    print("start run federated")

    for i in range(round):
        s_time = time.time()
        local_weight_list = []

        for u in range(user_number):
            model = build_model()
            print_train_info(i, u)
            server_weight = server.get_weight()

            if server_weight is not None:
                model.set_weights(server_weight)

            td, tl = make_split_train_data_by_number(u, size=1000)
            model.fit(td, tl, batch_size=batch_size, epochs=epochs, verbose=0)
            local_weight = model.get_weights()

            local_weight_list.append(local_weight)

        server.update_weight(local_weight_list)

        acc, loss = predict_part(i)
        round_result_acc.append(acc)
        round_result_loss.append(loss)
        round_result_time.append((time.time() - s_time))
        total_time.append((time.time() - start_time))

    make_graph(round_result_acc, round_result_loss)
    make_graph(round_result_time, total_time)

    print("end federated")


# %%
def predict_federated():
    model = build_model()
    model.set_weights(server.get_weight())
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    # print("test acc :{}, test loss :{}".format(test_acc, test_loss))

    result = model.predict(test_images)
    result = np.argmax(result, axis=1)

    cm = confusion_matrix(test_labels, result)
    print(cm)
    acc = accuracy_score(test_labels, result)
    print("acc : {}".format(acc))


# %%
def print_train_info(round=0, user_index=0):
    print("==========")
    print("round : {} , user : {}".format(round + 1, user_index + 1))
    print("==========")


# %%
def predict_part(round):
    model = build_model()
    model.set_weights(server.get_weight())

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    # print("test acc :{}, test loss :{}".format(test_acc, test_loss))

    '''
    result = model.predict(test_images)
    result = np.argmax(result, axis=1)
    acc = accuracy_score(test_labels, result)
    '''

    return test_acc, test_loss


# %%
start_time = time.time()
run_federated(10, 10, epochs=5, batch_size=10)
print("total time : {}".format(time.time() - start_time))

# %%
model = build_model()
model.set_weights(server.get_weight())

y_pred = result.predict(test_images).argmax(axis=1)
# y_actual = np.asarray(test_labels.argmax(axis=1)).reshape(len(test_labels))
# print(metrics.classification_report(y_actual, y_pred))

# %%    잘못된 분류 테스트
model = build_model()
model.set_weights(server.get_weight())

t_images, t_labels = make_split_test_data_by_number(5)

y_pred = model.predict(t_images)
y_pred = np.argmax(y_pred, axis=1)
print("label : ", t_labels)
print("predict : " , y_pred)
cm = confusion_matrix(t_labels, y_pred)
print(cm)
acc = accuracy_score(t_labels, y_pred)
print("acc : {}".format(acc))



# %%
print("None : ", metrics.f1_score(test_labels, y_pred, average=None))
print("micro : ", metrics.f1_score(test_labels, y_pred, average='micro'))
print("macro : ", metrics.f1_score(test_labels, y_pred, average='macro'))
print("weighted : ", metrics.f1_score(test_labels, y_pred, average='weighted'))
# print(metrics.f1_score(test_labels, y_pred, average='samples'))

# %%
server.init_weight()

# %%
with h5py.File("mnist_test/model/fl-nn-20191020.h5", 'w') as hf:
    for n, d in enumerate(server.get_weight()):
        hf.create_dataset(name='dataset{:d}'.format(n), data=d)


# %%
def weight_save(name):
    model.save_weights(name)


model = build_model()
model.set_weights(server.get_weight())

weight_save("mnist_test/model/fl-cnn-20191021.h5")


# %%
def load_weight(name):
    model = build_model()
    model.load_weights(name)
    return model


# %%
model = load_weight("mnist_test/model/fl-cnn-20191021.h5")

# %%
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("acc : ", test_acc, test_loss)

# %%
'''
    분류된 index image 리스트에서 (train_index_list) 특정 index에서 size갯수만큼 랜덤하게 뽑아 
    분할된 이미지 리스트를 만듦, sklearn의 mnist_test_split()과 유사한 기능  
'''


def make_split_train_data_by_number(index_number, size=600):
    random_index = np.random.randint(0, high=len(train_index_list[index_number]), size=size)

    s_train_image = []
    s_train_label = []
    for v in random_index:
        s_train_image.append(train_images[train_index_list[index_number][v]])
        s_train_label.append(train_labels[train_index_list[index_number][v]])
    return np.array(s_train_image), np.array(s_train_label)

# %%
def make_split_test_data_by_number(index_number):
    random_index = np.random.randint(0, high=len(test_index_list[index_number]), size = len(test_labels))

    s_test_image = []
    s_label_label = []
    for v in random_index:
        s_test_image.append(test_images[test_index_list[index_number][v]])
        s_label_label.append(test_labels[test_index_list[index_number][v]])
    return np.array(s_test_image), np.array(s_label_label)



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



# %%
def make_graph(acc_list=[], loss_list=[], round_result_time=[], total_time=[]):
    plt.plot(acc_list, 'r', label="CNN")
    # plt.plot(loss_list, '')
    plt.plot(round_result_time, 'b', label="FL-CNN")
    #plt.plot(total_time, 'b')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

    plt.title("Epochs for CNN", fontsize=10)
    plt.xlabel("Rounds for FL", fontsize=10)

    plt.legend()
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.show()


# %%
make_graph(acc_list=round_result_acc, loss_list=round_result_loss)
make_graph(round_result_time=round_result_time, total_time=total_time)

# %%
weight = server.get_weight()
print(weight.shape)

# %%
fl_model = load_weight("mnist_test/model/fl-cnn-20191021.h5")

# %%
result = fl_model.predict(test_images)
result = np.argmax(result, axis=-1)


f1 = f1_score(test_labels, result, average="macro")
print("Test f1 score : %s "% f1)

acc = accuracy_score(test_labels, result)
print("Test accuracy score : %s "% acc)

cm = confusion_matrix(test_labels, result)
print(cm)

measure2 = precision_recall_fscore_support(test_labels, result)
print("Test measure 2 : {}".format(measure2))

measure3 = classification_report(test_labels, result)
print("Test measure 3 : \n{}".format(measure3))

# %%
fl_result_list = 	[
0.098	,
0.098	,
0.156	,
0.369	,
0.380	,
0.473	,
0.471	,
0.494	,
0.518	,
0.690	,
0.620	,
0.614	,
0.612	,
0.587	,
0.686	,
0.651	,
0.655	,
0.604	,
0.670	,
0.651	,
0.702	,
0.733	,
0.700	,
0.668	,
0.654	,
0.677	,
0.743	,
0.696	,
0.729	,
0.751	,
0.787	,
0.711	,
0.731	,
0.718	,
0.713	,
0.702	,
0.742	,
0.721	,
0.715	,
0.667	,
0.728	,
0.765	,
0.764	,
0.769	,
0.804	,
0.799	,
0.803	,
0.797	,
0.822	,
0.777	,
0.789	,
0.803	,
0.822	,
0.784	,
0.767	,
0.745	,
0.785	,
0.787	,
0.819	,
0.840	,
0.849	,
0.830	,
0.829	,
0.812	,
0.782	,
0.820	,
0.801	,
0.817	,
0.851	,
0.816	,
0.827	,
0.760	,
0.783	,
0.740	,
0.833	,
0.782	,
0.800	,
0.735	,
0.800	,
0.789	,
0.786	,
0.813	,
0.849	,
0.829	,
0.862	,
0.843	,
0.822	,
0.781	,
0.766	,
0.767	,
0.863	]


# %%

make_graph(acc_list= normal_cnn.history.history["accuracy"], round_result_time=fl_result_list)


