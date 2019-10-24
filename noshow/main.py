# %%
'''
https://www.kaggle.com/joniarroba/noshowappointments

!!!
https://www.kaggle.com/belagoesr/predicting-no-show-downsampling-approach-with-rf
'''

# %%
import os
from datetime import datetime
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder

import time

import Base.BaseServer as BaseServer

le = LabelEncoder()
server = BaseServer.BaseServer.instance()

from sklearn import metrics
import matplotlib.pyplot as plt


# %%
def map_waiting_interval_to_days(x):
    if x == 0:
        return 'Less than 15 days'
    elif x > 0 and x <= 2:
        return 'Between 1 day and 2 days'
    elif x > 2 and x <= 7:
        return 'Between 3 days and 7 days'
    elif x > 7 and x <= 31:
        return 'Between 7 days and 31 days'
    else:
        return 'More than 1 month'

# %%
def map_age(x):
    if x < 12:
        return 'Child'
    elif x > 12 and x < 18:
        return 'Teenager'
    elif x >= 20 and x < 25:
        return 'Young Adult'
    elif x >= 25 and x < 60:
        return 'Adult'
    else:
        return 'Senior'

# %% Data processing
def processing_data(data):
    d = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    data['mapped_AppointmentDay'] = data['AppointmentDay'].map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
    data['mapped_ScheduledDay'] = data['ScheduledDay'].map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
    data['waiting_interval'] = abs(data['mapped_ScheduledDay'] - data['mapped_AppointmentDay'])
    data['waiting_interval_days'] = data['waiting_interval'].map(lambda x: x.days)
    data['waiting_interval_days'] = data['waiting_interval_days'].map(lambda x: map_waiting_interval_to_days(x))

    data['ScheduledDay_month'] = data['mapped_ScheduledDay'].map(lambda x: x.month)
    data['ScheduledDay_day'] = data['mapped_ScheduledDay'].map(lambda x: x.day)
    data['ScheduledDay_weekday'] = data['mapped_ScheduledDay'].map(lambda x: x.weekday())
    data['ScheduledDay_weekday'] = data['ScheduledDay_weekday'].replace(d)

    data['AppointmentDay_month'] = data['mapped_AppointmentDay'].map(lambda x: x.month)
    data['AppointmentDay_day'] = data['mapped_AppointmentDay'].map(lambda x: x.day)
    data['AppointmentDay_weekday'] = data['mapped_AppointmentDay'].map(lambda x: x.weekday())
    data['AppointmentDay_weekday'] = data['AppointmentDay_weekday'].replace(d)

    data['No-show'] = data['No-show'].replace({'Yes': 1, 'No': 0})

    missed_appointment = data.groupby('PatientId')['No-show'].sum()
    missed_appointment = missed_appointment.to_dict()
    data['missed_appointment_before'] = data.PatientId.map(lambda x: 1 if missed_appointment[x] > 0 else 0)
    data['mapped_Age'] = data['Age'].map(lambda x: map_age(x))
    data['Gender'] = data['Gender'].replace({'F': 0, 'M': 1})
    data['haveDisease'] = data.Alcoholism | data.Handcap | data.Diabetes | data.Hipertension

    data = data.drop(columns=['waiting_interval', 'AppointmentDay', 'ScheduledDay',
                              'PatientId', 'Age', 'mapped_ScheduledDay',
                              'mapped_AppointmentDay', 'AppointmentID',
                              'Alcoholism', 'Handcap', 'Diabetes', 'Hipertension'])
    return data

# %%
def one_hot_encode(data):
    return pd.get_dummies(data)


# %%
df_train = pd.read_csv('noshow/KaggleV2-May-2016.csv')

processed_data = processing_data(df_train)
print(processed_data.head())

encoded_data = one_hot_encode(processed_data)
print(encoded_data.head())
'''
binary_features = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Gender']
target = ['No-show']
categorical = ['Neighbourhood', 'Handcap']
numerical = ['Age']
dates = ['AppointmentDay', 'ScheduledDay']
Ids = ['PatientId', 'AppointmentID']
'''

# %%
df_train['AppointmentDay'] = pd.to_datetime(df_train['AppointmentDay'])
df_train['ScheduledDay'] = pd.to_datetime(df_train['ScheduledDay'])

df_train['waiting_days'] = (df_train['AppointmentDay'] - df_train['ScheduledDay']).dt.days

df_train = df_train[(df_train['waiting_days'] >= -1) & (df_train['waiting_days'] <= 100)]

df_train.Gender = df_train['Gender'].map({"F": 0, "M": 1})
df_train['No-show'] = df_train['No-show'].map({"No": 0, "Yes": 1})

less_than_100 = ['MORADA DE CAMBURI', 'PONTAL DE CAMBURI', 'ILHA DO BOI', 'ILHA DO FRADE',
                 'AEROPORTO', 'ILHAS OCEÂNICAS DE TRINDADE', 'PARQUE INDUSTRIAL']

#df_train.loc[df_train.Neighbourhood.isin(less_than_100), 'Neighbourhood'] = "OTHERS"

df_train.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1, inplace=True)

#le.fit(df_train['Neighbourhood'].drop_duplicates())
#df_train['Neighbourhood'] = le.transform(df_train['Neighbourhood'])

# %%
#y_train = df_train['No-show']
y_train = encoded_data['No-show']
#X_train = df_train.drop('No-show', axis=1)
X_train = encoded_data.drop('No-show', axis=1)

#train_stats = X_train.describe()
#train_stats = train_stats.transpose()

#X_train = (X_train - train_stats['mean'] ) / train_stats['std']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

result = [x for x in y_train if x == 1]
print("total train size : ", len(y_train), " / no :" , (len(y_train) - len(result)), " : yes " ,len(result))

result = [x for x in y_val if x == 1]
print("total test size : ", len(y_val), " / no :" ,  (len(y_val) - len(result)), " : yes " ,len(result))

# %%
def print_cm(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    print("accuracy : {}".format(((cm[0][0] + cm[1][1]) / len(X_val)) * 100))

# %% roc curve
def print_roc_curve(y_test, y_predict):
    y_true = np.array(y_test)
    y_probas = y_predict

    print(len(y_true), len(y_probas) , y_true.shape, y_probas.shape)


    fpr, tpr, thresholds = roc_curve(y_true, y_probas)
    roc_auc = auc(fpr, tpr)
    print(fpr, tpr, roc_auc)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# %%
def make_acc_graph(acc_list=[]):
    plt.plot(acc_list, "r",  label="acc")
    plt.legend()
    #plt.xlim(xmin = 0)
    #plt.ylim(ymin = 0)

    plt.show()

# %%
def make_time_graph(round_result_time, total_time):
    plt.plot(round_result_time, 'g')
    plt.plot(total_time, 'b')

    plt.legend()
    plt.show()

# %%
def build_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_dim=112),
        tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])


    return model

model = build_model()

# %%
start_time = time.time()
model.fit(X_train, y_train, batch_size=64, epochs=20)
print("total time : {}".format(time.time()-start_time))

# %%
make_acc_graph(model.history.history['accuracy'])
#make_time_graph(round_result_time, total_time)
# %%    nn
print(len(y_val))
y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5)

print_cm(y_val, y_pred)

#print_roc_curve(y_val, y_pred)

# %%    random forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_val)
print("result count : ", np.count_nonzero(y_pred == 0), np.count_nonzero(y_pred == 1))
print_cm(y_val, y_pred)


# %% GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_val)
print("result count : ", np.count_nonzero(y_pred == 0), np.count_nonzero(y_pred == 1))
print_cm(y_val, y_pred)

# %% Support Vector
from sklearn import svm
clf = svm.SVC(gamma=0.001, C = 1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print("result count : ", np.count_nonzero(y_pred == 0), np.count_nonzero(y_pred == 1))
print_cm(y_val, y_pred)

# %%
def print_train_info(round = 0, user_index = 0):
    print("==========")
    print("round : {} , user : {}".format(round+1, user_index+1))
    print("==========")

# %%
round_result_acc = []
round_result_acc.append(0)
round_result_loss = []
round_result_time = []
total_time = []
def run_federate(user_number = 3, round_number =2, epoch = 5, batch_size = 10):
    train_list = np.array_split(X_train, user_number)
    test_list = np.array_split(y_train, user_number)
    for r in range(round_number):
        s_time = time.time()
        local_weight_list = []

        for user in range(user_number):
            local_model = build_model()
            print_train_info(r, user)
            server_weight = server.get_weight()

            if server_weight is not None:
                local_model.set_weights(server_weight)

            local_model.fit(train_list[user], test_list[user], epochs=epoch, batch_size=batch_size, verbose=0)
            local_weight_list.append(local_model.get_weights())
        server.update_weight(local_weight_list)

        acc, loss = predict_part(X_val, y_val)
        round_result_acc.append(acc)
        round_result_loss.append(loss)
        round_result_time.append((time.time()-s_time))
        total_time.append((time.time()-start_time))

    print("federate end")
    make_acc_graph(round_result_acc)

    #predict_federate(train_list[-1], test_list[-1], X_val, y_val)

# %%
def non_federate(x_train, y_train, x_test, y_test ):
    print("non_federate validation start")
    model = build_model()
    #model.fit(x_train, y_train, epochs=5, verbose=0)
    result = model.predict(x_test)
    result = (result > 0.5)
    print_cm(y_test, result)
    print("non_federate validation end\n")



# %%
def predict_federate(x_train, y_train, x_test, y_test):
    print("federate validation start")
    model = build_model()
    model.set_weights(server.get_weight())
    result = model.predict(x_test)
    result = (result > 0.5)
    print_cm(y_test, result)
    print("federate validation end")

    #print_roc_curve(y_test, result)

# %%
def predict_part(x_test, y_test):
    model = build_model()
    model.set_weights(server.get_weight())

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    #print("test acc :{}, test loss :{}".format(test_acc, test_loss))

    return test_acc, test_loss


# %%
start_time = time.time()
run_federate(user_number=10, round_number=50, batch_size=16, epoch=5)
print("total time : {}".format(time.time()-start_time))

# %%
make_acc_graph(round_result_acc)
make_time_graph(round_result_time, total_time)

# %%
predict_part(X_val, y_val)

# %%
model = build_model()
model.set_weights(server.get_weight())
result = model.predict(X_val)
result = (result > 0.5)
print_cm(y_val, result)


# %%
server.init_weight()

# %%
def weight_save(name):
    model.save_weights(name)

weight_save("noshow/model/fl-balance-20191019.h5")

# %%
def weight_load(name):
    loaded_model = model.load_weights(name)
    return loaded_model
