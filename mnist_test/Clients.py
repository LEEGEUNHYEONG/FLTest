# %%
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import matplotlib.ticker as mticker
import time
from mnist_test.Model import TestModel
# %%
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
def print_round(number):
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('--------------------------------------- round : ', number)
    print('--------------------------------------- round : ', number)
    print('--------------------------------------- round : ', number)
    print('--------------------------------------- round : ', number)
    print('--------------------------------------- round : ', number)
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')
    print('----------------------------------------------------------------------')

# %%
'''
    mnist의 특정 숫자만 federate 학습 하도록 함 
'''


def learning_federated_number(value=0):
    ti, tl = make_sublist(train_index_list, train_images, train_labels, value)
    m1 = TestModel()
    m1.set(ti, tl, server.get_weight())
    server.update_value(m1.get_weight())


# %%
def learning_federated_split_number(value):
    s_train_image, s_train_label = make_split_train_data_by_number(value)
    model = TestModel()
    model.set(s_train_image, s_train_label, server.get_weight(), batch_size=10)
    server.update_value(model.get_weight())

# %%
def evaluate_number(value=-1):
    m1 = TestModel()
    m1.set(train_images, train_labels, batch_size=128, epoch=12)
    if value == -1:
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

def make_graph(acc_list, loss_list):
    plt.plot(acc_list, 'r')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

    #plt.plot(loss_list, 'g')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.show()


# %%
def run_federate(user_number=1, round=1, batch_size = 10, epoch = 5):
    print("start federate")
    acc_list = []
    acc_list.append(0)
    loss_list = []
    local_model = TestModel()

    for i in range(round):
        server_weight = server.get_weight2()
        print_round(i)
        local_weight_list = []  # local weight list
        local_acc_list = []        # fit 결과의 acc 저장, 각 라운드의 마지막 유저의 acc 만 저장
        local_loss_list = []        # fit 결과의 acc 저장, 각 라운드의 마지막 유저의 acc 만 저장
        for u in range(user_number):
            #ti, tl = make_split_train_data()
            ti, tl = make_split_train_data_by_number(0)
            hist, local_weight = local_model.set(ti, tl, server_weight, batch_size=batch_size, epoch=epoch)
            local_weight_list.append(local_weight)

            ti, tl = make_split_train_data_by_number(9)
            hist, local_weight = local_model.set(ti, tl, server_weight, batch_size=10, epoch=5)
            local_weight_list.append(local_weight)

            #local_acc_list.append(hist.history['acc'][-1:])
            #local_loss_list.append(hist.history['loss'][-1:])
            local_weight_list.append(local_weight)

        server.update_weight2(local_weight_list)

        result = evaluate_federated_number(-1)
        acc_list.append(result[1] * 100)
        # 매 라운드 종료 시 evaluate 시행
        #acc_list.append(local_acc_list[-1:])
        #loss_list.append(local_loss_list[-1:])

    acc_list = np.array(acc_list)
    #acc_list = acc_list.ravel()
    #loss_list = np.array(loss_list)
    #loss_list = loss_list.ravel()

    make_graph(acc_list, loss_list)


# %%
'''
    서버에서 fed_weight를 받아와 특정 숫자로 로컬 테스트 진행
'''
def evaluate_federated_number(value=-1):
    m1 = TestModel()
    s_image, s_label = make_split_train_data()
    m1.set(s_image, s_label, weights=server.get_weight2())
    if value == -1:
        # s_test_image, s_test_label = make_split_test_data()
        # m1.local_evaluate(s_test_image, s_test_label)
        result = m1.local_evaluate(test_images, test_labels)
    else:
        ti, tl = make_sublist(test_index_list, test_images, test_labels, value)
        result = m1.local_evaluate(ti, tl)

    return result


# %%
start_time = time.time()
run_federate(user_number=10, round = 200, batch_size=10, epoch=20 )
print("time : {}".format(time.time()-start_time))


# %%
#evaluate_federated_number()

# %%
#evaluate_number()

# %%
#server.clear_weight()
