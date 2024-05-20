''' importing libraries '''

import sys
sys.path.append('libraries')

import pandas as pd
import joblib

# loading the model and preprocessors

encoder = joblib.load('pickle_files/encoder.pkl')
scaler = joblib.load('pickle_files/scaler.pkl')
model = joblib.load('pickle_files/model.pkl')

# processing the input

def process_input(input_data):
    feature_names = ['T1C1', 'T1C2', 'T1C3', 'T1C4', 'T1C5', 'T2C1', 'T2C2', 'T2C3', 'T2C4', 'T2C5']
    feature_values = pd.DataFrame(input_data, columns=feature_names).values
    encoding = encoder.transform(feature_values)
    features = pd.DataFrame.sparse.from_spmatrix(encoding, columns=encoder.get_feature_names_out(feature_names))
    return features

# making predictions

def model_predict(input_data):
    data = process_input(input_data)
    predictions = model.predict(data)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).ravel() 
    return predictions

''' custom input '''

input_data = [input().split(',')]
predictions = model_predict(input_data)

print("Predicted Gold Difference:", int(predictions[0]))
if predictions[0] > 0:
    print("Predicted Winner: Team 1")
else:
    print("Predicted Winner: Team 2")