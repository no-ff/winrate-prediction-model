''' importing libraries '''

import sys
sys.path.append('libraries')

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score

''' hyperparameters tuning '''

parameters = {
    'regressor__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'regressor__max_depth': [None, 10, 20, 30, 40, 50],
    'regressor__max_features': [None, 'sqrt', 'log2'],
    'regressor__min_samples_leaf': [1, 2, 4, 8, 10],
    'regressor__min_samples_split': [2, 5, 10, 20, 30],
    'regressor__n_estimators': [100, 200, 300, 500, 1000]
}

''' Current best hyperparameters '''
best_parameters = {
    'criterion': 'friedman_mse', 
    'max_depth': 20, 
    'max_features': None, 
    'min_samples_leaf': 8, 
    'min_samples_split': 20, 
    'n_estimators': 100
}

''' main model trainer '''

def train():

    ''' preprocessing ''' 

    df = pd.read_csv('data/csv_files/cleaned_output.csv')
    # print(df.head())

    # features and target
    features = df[['T1C1', 'T1C2', 'T1C3', 'T1C4', 'T1C5', 'T2C1', 'T2C2', 'T2C3', 'T2C4', 'T2C5', 'W1', 'W2', 'W3', 'W4', 'W5']]
    categorical_cols = ['T1C1', 'T1C2', 'T1C3', 'T1C4', 'T1C5', 'T2C1', 'T2C2', 'T2C3', 'T2C4', 'T2C5']
    numerical_cols = ['W1', 'W2', 'W3', 'W4', 'W5']
    target = df['GD']

    # initlize encoder and scaler
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = MinMaxScaler()
    output_scaler = MinMaxScaler(feature_range=(-1, 1))

    # define the processor
    processor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ]
    )

    # define the model
    predictor = Pipeline(
        steps=[
            ('preprocessor', processor),
            ('regressor', RandomForestRegressor(**best_parameters)) # using best parameters
        ]
    )

    # split the data
    X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    # scaling the target
    y_train = output_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test = output_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)

    ''' training '''

    # finding the best parameters
    # random_search = RandomizedSearchCV(estimator=predictor, param_distributions=parameters, n_iter=500, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(r2_score))
    # random_search.fit(X_train, y_train)
    # print("Best parameters: ", random_search.best_params_)
    # predictor = random_search.best_estimator_
    
    predictor.fit(X_train, y_train)

    # evaluation
    y_pred = predictor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
    print(r2_score(y_test, y_pred))

    # saving the model
    joblib.dump(predictor, 'models/model_v1.pkl')

train()