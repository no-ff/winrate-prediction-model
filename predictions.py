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
    return predictions

# calculating win percentage

def calculate_percentage(p):
    print(f"Predicted win percentage for team 1: {int((p + 1) / 2 * 100)}%")
    if p > 0:
        print("Predicted winner: team 1")
    else:
        print("Predicted winner: team 2")

''' custom input '''

try:
    input_data = [input().split(',')] 
    # example: "Irelia,Darius,Xerath,Anivia,Sejuani,Diana,Jayce,Maokai,Neeko,Kaisa"
    predictions = model_predict(input_data)
    calculate_percentage(predictions[0])

except Exception as e:
    print("Check your input again!")