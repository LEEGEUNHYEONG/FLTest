'''
    https://www.kaggle.com/thebrownviking20/intro-to-keras-with-breast-cancer-data-ann
'''
# %%
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import Base.BaseServer as BaseServer
import numpy as np

# %%
server = BaseServer.BaseServer.instance()
data = pd.read_csv('BreastCancerWisconsin/data.csv')
del data['Unnamed: 32']

# %%
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("train size : {}, test size : {}".format(len(X_train), len(X_test)))

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# %%
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=16, kernel_initializer='uniform', activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])

    # original
    '''
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    '''

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# %%
model = build_model()

# %%
# original
#model.fit(X_train, y_train, batch_size=100, epochs=150)

model.fit(X_train, y_train, batch_size=10, epochs=1)

# %%
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("accuracy : {}".format(((cm[0][0] + cm[1][1]) / len(X_test)) * 100))

sns.heatmap(cm, annot=True)
plt.show()

# %%
'''
for i in range(len(y_pred)):
    print("{} >>> {}".format(y_test[i], y_pred[i]))
'''
# %%
def run_federate(user_number=3, round_number=2, epoch=20, batch_size=10):
    train_list = np.array_split(X_train, user_number+1)
    test_list = np.array_split(y_train, user_number+1)
    print("federate start : user number : {}, total size : {}, each size : {}".format(user_number, len(X_train),len(train_list[0]) ))

    local_model = build_model()
    for r in range(round_number):
        local_weight_list = []
        print('---------')
        print("round : ", (r+1))
        print('---------')
        for user in range(user_number):
            server_weight = server.get_weight()
            if server_weight is not None:
                local_model.set_weights(server_weight)
            local_model.fit(train_list[user], test_list[user], epochs=epoch, batch_size=batch_size, verbose=0)
            local_weight_list.append(local_model.get_weights())

        server.update_weight(local_weight_list)
    print("federate end")

    predict_federate(train_list[-1], test_list[-1], X_test, y_test)

# %%
def predict_federate(x_train, y_train, x_test, y_test):
     model = build_model()
     model.set_weights(server.get_weight())
     model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=0)
     result = model.predict(x_test)
     result = (result > 0.5)
     show_confusion_matrix(y_test, result)


# %%
def show_confusion_matrix(y_test, predict):
    cm = confusion_matrix(y_test, predict)
    sns.heatmap(cm, annot=True)
    plt.show()
    print(cm)
    print("accuracy : ", ((cm[0][0] + cm[1][1]) / len(y_test)) * 100)


# %%
run_federate(user_number=5, round_number=2, batch_size=10, epoch=20)

# %%
predict_federate(X_train, y_train, X_test, y_test)

# %%
server.init_weight()



