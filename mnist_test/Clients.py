# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.ticker as mticker
import time
import tensorflow as tf
from tensorflow import keras
import Base.BaseServer as Server
import pydot
import graphviz
# %%
server = Server.BaseServer.instance()

'''
#   그래픽카드 사용 확인 
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

np.random.seed(42)

# %%
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255, test_images / 255

#   cnn인 경우 reshape 필요
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1,  28, 28, 1))
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
       tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
       tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
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
keras.utils.plot_model(build_model(), "model.png", show_shapes = True)

# %%

def train():
    model = build_model()
    result = model.fit(train_images, train_labels, batch_size=64, epochs=20)

    score = model.evaluate(test_images, test_labels, verbose=2)
    print("test acc :{}, test loss :{}".format(score[1], score[0]))
    return model

start_time = time.time()
result = train()
print("time : {}".format(time.time()-start_time))

# %%
y_pred = result.predict(test_images).argmax(axis=1)
#y_actual = np.asarray(test_labels.argmax(axis=1)).reshape(len(test_labels))
#print(metrics.classification_report(y_actual, y_pred))

# %%
print("None : ", metrics.f1_score(test_labels, y_pred, average=None))
print("micro : ", metrics.f1_score(test_labels, y_pred, average='micro'))
print("macro : ", metrics.f1_score(test_labels, y_pred, average='macro'))
print("weighted : ", metrics.f1_score(test_labels, y_pred, average='weighted'))
#print(metrics.f1_score(test_labels, y_pred, average='samples'))


# %%
round_result_acc = []
round_result_loss = []
round_result_time = []
total_time = []
def run_federated(user_number = 1, round = 1, batch_size = 5, epochs = 10):
    print("start run federated")
    acc_list = []
    loss_list = []
    round_result_list = []

    for i in range(round):
        s_time = time.time()
        local_weight_list = []

        for u in range(user_number):
            model = build_model()
            print_train_info(i, u)
            server_weight = server.get_weight()

            if server_weight is not None:
                model.set_weights(server_weight)

            td, tl = make_split_train_data_by_number(u, size= 1000)
            model.fit(td, tl, batch_size=batch_size, epochs=epochs, verbose=0)
            local_weight_list.append(model.get_weights())

        server.update_weight(local_weight_list)


        acc, loss = predict_part(i)
        round_result_acc.append(acc)
        round_result_loss.append(loss)
        round_result_time.append((time.time()-s_time))
        total_time.append((time.time()- start_time))


    #print("acc : {} , loss : {}".format(round_result_acc, round_result_loss))
    make_graph(round_result_acc, round_result_loss)
    make_graph(round_result_time, total_time)
    #predict_federated()

# %%
def predict_federated():
    model = build_model()
    model.set_weights(server.get_weight())
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #print("test acc :{}, test loss :{}".format(test_acc, test_loss))

    result = model.predict(test_images)
    result = np.argmax(result, axis = 1)

    cm = confusion_matrix(test_labels, result)
    print(cm)
    acc = accuracy_score(test_labels, result)
    print("acc : {}".format(acc))

# %%
def print_train_info(round = 0, user_index = 0):
    print("==========")
    print("round : {} , user : {}".format(round+1, user_index+1))
    print("==========")

# %%
def predict_part(round):
    model = build_model()
    model.set_weights(server.get_weight())

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    #print("test acc :{}, test loss :{}".format(test_acc, test_loss))

    '''
    result = model.predict(test_images)
    result = np.argmax(result, axis=1)
    acc = accuracy_score(test_labels, result)
    '''

    return test_acc, test_loss


# %%
start_time = time.time()
run_federated(5, 5, epochs=5, batch_size=10)
print("total time : {}".format(time.time()-start_time))

# %%
model = build_model()
model.set_weights(server.get_weight())

y_pred = result.predict(test_images).argmax(axis=1)
#y_actual = np.asarray(test_labels.argmax(axis=1)).reshape(len(test_labels))
#print(metrics.classification_report(y_actual, y_pred))

# %%
print("None : ", metrics.f1_score(test_labels, y_pred, average=None))
print("micro : ", metrics.f1_score(test_labels, y_pred, average='micro'))
print("macro : ", metrics.f1_score(test_labels, y_pred, average='macro'))
print("weighted : ", metrics.f1_score(test_labels, y_pred, average='weighted'))
#print(metrics.f1_score(test_labels, y_pred, average='samples'))

# %%
server.init_weight()

# %%
with h5py.File("mnist_test/model/fl-nn-20191020.h5", 'w') as hf:
    for n, d in enumerate(server.get_weight()):
        hf.create_dataset(name = 'dataset{:d}'.format(n), data=d)


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
    plt.plot(acc_list, 'r')
    #plt.plot(loss_list, '')
    plt.plot(round_result_time, 'g')
    plt.plot(total_time, 'b')
    #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

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


