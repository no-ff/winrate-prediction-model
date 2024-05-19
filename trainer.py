# importing libraries
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

# Preprocessing
df = pd.read_csv('cleaned_output.csv')
print(df.head())

feature_columns = df.columns[:-2]
target_column = df.columns[-1]