''' importing libraries '''

import sys
sys.path.append('libraries')

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

'''Preprocessing'''

df = pd.read_csv('cleaned_output.csv')
# print(df.head())

features = df.columns[:-1]
target = df.columns[-1]

team_1 = features[:5]
team_2 = features[5:]

# initlize encoder and scaler
encoder = OneHotEncoder(sparse_output=True)
scaler = MinMaxScaler(feature_range=(-1, 1))

# encode the teams seperately
encoded_team_1 = encoder.fit_transform(df[team_1])
encoded_team_1_df = pd.DataFrame.sparse.from_spmatrix(encoded_team_1, columns=encoder.get_feature_names_out(team_1))
encoded_team_2 = encoder.fit_transform(df[team_2])
encoded_team_2_df = pd.DataFrame.sparse.from_spmatrix(encoded_team_2, columns=encoder.get_feature_names_out(team_2))

# scale the gold difference
scaled_GD = scaler.fit_transform(df[[target]])
scaled_GD_df = pd.DataFrame(scaled_GD, columns=[target])

# replace the original categorical columns with the encoded columns in the dataframe
df.drop(df.columns, axis=1, inplace=True)
df = pd.concat([encoded_team_1_df, encoded_team_2_df, scaled_GD_df, df], axis=1)
print(df.head())

''' Training the model '''

