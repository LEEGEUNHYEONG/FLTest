# %%
'''
https://www.kaggle.com/joniarroba/noshowappointments
'''
# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import scipy as sp
from sklearn.model_selection import RandomizedSearchCV
from functools import partial
from sklearn.metrics import confusion_matrix

le = LabelEncoder()

# %%
df_train = pd.read_csv('noshow/KaggleV2-May-2016.csv')

binary_features = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'Gender']
target = ['No-show']
categorical = ['Neighbourhood', 'Handcap']
numerical = ['Age']
dates = ['AppointmentDay', 'ScheduledDay']
Ids = ['PatientId', 'AppointmentID']

# %%
df_train['AppointmentDay'] = pd.to_datetime(df_train['AppointmentDay'])
df_train['ScheduledDay'] = pd.to_datetime(df_train['ScheduledDay'])

df_train['waiting_days'] = (df_train['AppointmentDay'] - df_train['ScheduledDay']).dt.days

df_train = df_train[(df_train['waiting_days'] >= -1) & (df_train['waiting_days'] <= 100)]

df_train.Gender = df_train['Gender'].map({"F": 0, "M": 1})
df_train['No-show'] = df_train['No-show'].map({"No": 0, "Yes": 1})

less_than_100 = ['MORADA DE CAMBURI', 'PONTAL DE CAMBURI', 'ILHA DO BOI', 'ILHA DO FRADE',
                 'AEROPORTO', 'ILHAS OCEÂNICAS DE TRINDADE', 'PARQUE INDUSTRIAL']

df_train.loc[df_train.Neighbourhood.isin(less_than_100), 'Neighbourhood'] = "OTHERS"

df_train.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)

le.fit(df_train['Neighbourhood'].drop_duplicates())
df_train['Neighbourhood'] = le.transform(df_train['Neighbourhood'])

# %%
y_train = df_train['No-show']
X_train = df_train.drop('No-show', axis=1)

train_stats = X_train.describe()
train_stats = train_stats.transpose()

X_train = (X_train - train_stats['mean'] ) / train_stats['std']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2)

result = [x for x in y_train if x == 1]
print(len(y_train), " : " , len(result))

result = [x for x in y_val if x == 1]
print(len(y_val), " : ", len(result))

# %%
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_dim=10),
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
        #tf.keras.layers.Dropout(0.25),
        #tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])


    return model

model = build_model()
# %%
model.fit(X_train, y_train, batch_size=64, epochs=5)

# %%
y_pred = model.predict(X_val)

result = [x for x in y_pred if x > 0.5]
print(len(result))

y_pred = (y_pred > 0.5)

print("result count : ", np.count_nonzero(y_pred == 0), np.count_nonzero(y_pred == 1))

cm = confusion_matrix(y_val, y_pred)
print(cm)
print("accuracy : {}".format(((cm[0][0] + cm[1][1]) / len(X_val)) * 100))

# %%    random forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_val)

cm = confusion_matrix(y_val, y_pred)
print(cm)
print("accuracy : {}".format(((cm[0][0] + cm[1][1]) / len(X_val)) * 100))

# %% GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_val)

cm = confusion_matrix(y_val, y_pred)
print(cm)
print("accuracy : {}".format(((cm[0][0] + cm[1][1]) / len(X_val)) * 100))

# %% Support Vector
from sklearn import svm
clf = svm.SVC(gamma=0.001, C = 1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

cm = confusion_matrix(y_val, y_pred)
print(cm)
print("accuracy : {}".format(((cm[0][0] + cm[1][1]) / len(X_val)) * 100))