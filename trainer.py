''' importing libraries '''

import sys
sys.path.append('libraries')

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

'''Preprocessing'''

df = pd.read_csv('cleaned_output.csv')
# print(df.head())

categorical_columns = df.columns[:-2]
numerical_column = df.columns[-2]

team_1 = categorical_columns[:5]
team_2 = categorical_columns[5:]

# initlize encoder and scaler
encoder = OneHotEncoder(sparse_output=True)
scaler = StandardScaler()

# encode the teams seperately
encoded_team_1 = encoder.fit_transform(df[team_1])
encoded_team_1_df = pd.DataFrame.sparse.from_spmatrix(encoded_team_1, columns=encoder.get_feature_names_out(team_1))
encoded_team_2 = encoder.fit_transform(df[team_2])
encoded_team_2_df = pd.DataFrame.sparse.from_spmatrix(encoded_team_2, columns=encoder.get_feature_names_out(team_2))

# scale the numerical data 
scaled_GD = scaler.fit_transform(df[[numerical_column]])
scaled_GD_df = pd.DataFrame(scaled_GD, columns=[numerical_column])

# replace the original categorical columns with the encoded columns in the dataframe
df = df.drop(categorical_columns, axis=1)
df = df.drop(numerical_column, axis=1)
df = pd.concat([encoded_team_1_df, encoded_team_2_df, scaled_GD_df, df], axis=1)
# print(df.head())

features = df.drop('Target', axis=1)
target = df['Target']

''' Training the model '''