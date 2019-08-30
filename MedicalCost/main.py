'''
    https://www.kaggle.com/hely333/eda-regression
'''
# %%    load csv
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Base import BaseFedClient

fedClient = BaseFedClient.BaseFedClient()

warnings.filterwarnings('ignore')
data = pd.read_csv('../FLTest/MedicalCost/insurance.csv')

# %%    데이터 확인
print(data.head())
print(data.isnull().sum())

# %%    value change by on-hot-encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# gender, female - 0, male - 1
le.fit(data.sex.drop_duplicates())
data.sex = le.transform(data.sex)

# is smoker, 1 - yes, 0 - no
le.fit(data.smoker.drop_duplicates())
data.smoker = le.transform(data.smoker)

# region, SW - 3 , SE - 2, NW -1, NE - 0
le.fit(data.region.drop_duplicates())
data.region = le.transform(data.region)

print(data)
# %%


# %%    LinearRegression
x = data.drop(['charges'], axis=1)
y = data.charges

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
lr = LinearRegression().fit(x_train, y_train)

print(len(x_train), len(x_test))

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test, y_test))

# %%    LinearRegression region delete
X = data.drop(['charges', 'region'], axis=1)
Y = data.charges

quad = PolynomialFeatures(degree=2)
x_quad = quad.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(x_quad, Y, random_state=0)

print(len(x_train), len(x_test))

plr = LinearRegression().fit(X_train, Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test, Y_test))

# %%    RandomForest
forest = RandomForestRegressor(n_estimators=100,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)
forest.fit(x_train, y_train)

forest_train_pred = forest.predict(x_train)
forest_test_pred = forest.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
    mean_squared_error(y_train, forest_train_pred),
    mean_squared_error(y_test, forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
    r2_score(y_train, forest_train_pred),
    r2_score(y_test, forest_test_pred)))

# %%
fedClient.test()


