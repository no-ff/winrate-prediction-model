''' importing libraries '''

import sys
sys.path.append('libraries')

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

''' preprocessing ''' 

df = pd.read_csv('cleaned_output.csv')
# print(df.head())

feature_names = df.columns[:-1]
feature_values = df[feature_names].values
target = df['GD'].values

# initlize encoder and scaler
encoder = OneHotEncoder(sparse_output=True)
scaler = MinMaxScaler(feature_range=(-1, 1))

# encode the teams seperately
encoding = encoder.fit_transform(feature_values)
features = pd.DataFrame.sparse.from_spmatrix(encoding, columns=encoder.get_feature_names_out(feature_names))

# print(features)
# print(target)

X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

scaler = MinMaxScaler(feature_range=(-1, 1))
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
scaler.fit(y_train)
y_train = scaler.transform(y_train).ravel()
y_test = scaler.transform(y_test).ravel()

# print(y_train)
# print(y_test)

''' training the model '''

regressor = RandomForestRegressor(n_estimators=1000, random_state=0)
regressor.fit(X_train, y_train)

''' evaluating the model '''

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

print(r2_score(y_test, y_pred))

''' saving the model '''

joblib.dump(regressor, 'pickle_files/model.pkl')
joblib.dump(scaler, 'pickle_files/scaler.pkl')
joblib.dump(encoder, 'pickle_files/encoder.pkl')