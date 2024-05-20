''' importing libraries '''

import sys
sys.path.append('libraries')

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score

from data_cleaner import clean_data

''' hyperparameters tuning '''

# parameters = {
#     'n_estimators': [100, 300, 500],
#     'max_features': [None, 'sqrt', 'log2'],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 2, 4, 8],
#     'criterion': ['poisson', 'absolute_error', 'friedman_mse', 'squared_error'],
# }

''' hyperparameters (current best) '''
parameters = {
    'n_estimators': 100,
    'max_features': 'sqrt',
    'max_depth': 20,
    'min_samples_split': 20,
    'min_samples_leaf': 1,
    'criterion': 'friedman_mse',
}

''' main model trainer '''

def train():

    ''' preprocessing ''' 

    # clean the data
    # clean_data()

    df = pd.read_csv('csv_files/cleaned_output.csv')
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

    ''' training '''

    # tuning the model
    # regressor = RandomForestRegressor()
    # grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(r2_score))
    # grid_search.fit(X_train, y_train)
    # print("Best parameters: ", grid_search.best_params_)
    # predictor = grid_search.best_estimator_

    # using the current best hyperparameters
    predictor = RandomForestRegressor(**parameters)
    predictor.fit(X_train, y_train)

    # evaluation
    y_pred = predictor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
    print(r2_score(y_test, y_pred))

    # saving the model
    joblib.dump(predictor, 'pickle_files/model.pkl')
    joblib.dump(scaler, 'pickle_files/scaler.pkl')
    joblib.dump(encoder, 'pickle_files/encoder.pkl')

train()