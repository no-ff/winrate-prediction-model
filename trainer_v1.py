''' importing libraries '''

import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score
from get_champion_weights import get_weights

''' hyperparameters tuning '''

parameters = {
    'regressor__n_estimators': np.random.randint(100, 1000, size=100),
    'regressor__max_depth': np.random.randint(2, 20, size=100),
    'regressor__learning_rate': np.random.uniform(0.01, 0.1, size=100),
    'regressor__subsample': np.random.uniform(0.6, 1.0, size=100),
    'regressor__colsample_bytree': np.random.uniform(0.6, 1.0, size=100),
    'regressor__gamma': np.random.uniform(0, 0.5, size=100),
    'regressor__min_child_weight': np.random.randint(1, 20, size=100)
}

''' main model trainer '''

<<<<<<< HEAD
def train_1():
=======
def load_data(filepath, new_data_format=True):
    if not new_data_format:
        return pd.read_csv(filepath)

    # otherwise, assume there is no header.
    df = pd.read_csv(filepath, header=None)
    df.columns = ['T1C1', 'T1C2', 'T1C3', 'T1C4', 'T1C5', 'T2C1', 'T2C2', 'T2C3', 'T2C4', 'T2C5', 'GD', 'WL', 'REGION', 'ELO']

    for champ in range(1,6):
        new_col=[]
        for i, row in df.iterrows():
            c1 = row[f'T1C{champ}']
            c2 = row[f'T2C{champ}']
            new_col.append(get_weight(c1, c2, champ-1))
        
        df.insert(len(df.columns), f'W{champ}', new_col, False)
    
    return df

def train():
>>>>>>> 8894ecc9f3c67e7c67567dae0ccaa6ded0d17956

    ''' preprocessing '''

<<<<<<< HEAD
    df = pd.read_csv('data/csv_files/cleaned_output.csv')
=======
    df = load_data('data/csv_files/100k.csv')
    # print(df.head())
>>>>>>> 8894ecc9f3c67e7c67567dae0ccaa6ded0d17956

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
<<<<<<< HEAD
            ('regressor', xgb.XGBRegressor()) # using best parameters
=======
            ('regressor', RandomForestRegressor()) # using best parameters
>>>>>>> 8894ecc9f3c67e7c67567dae0ccaa6ded0d17956
        ]
    )

    # scaling the target
    target = target.values.reshape(-1, 1)
    target = output_scaler.fit_transform(target).ravel()

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=0)

    ''' training '''

    # finding the best parameters
<<<<<<< HEAD
    random_search = RandomizedSearchCV(estimator=predictor, param_distributions=parameters, n_iter=500, cv=3, n_jobs=-1, verbose=1, scoring=make_scorer(r2_score))
    random_search.fit(X_train, y_train)
    print("Best parameters: ", random_search.best_params_)
    predictor = random_search.best_estimator_

=======
    random_search = RandomizedSearchCV(estimator=predictor, param_distributions=parameters, n_iter=2, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(r2_score))
    random_search.fit(X_train, y_train)
    print("Best parameters: ", random_search.best_params_)
    predictor = random_search.best_estimator_
    
>>>>>>> 8894ecc9f3c67e7c67567dae0ccaa6ded0d17956
    predictor.fit(X_train, y_train)

    # Evaluation
    y_pred = predictor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
    print(r2_score(y_test, y_pred))

    # saving the model
    joblib.dump(predictor, 'model_v1.pkl')

<<<<<<< HEAD
train_1()
=======
train()
>>>>>>> 8894ecc9f3c67e7c67567dae0ccaa6ded0d17956
