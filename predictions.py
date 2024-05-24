''' importing libraries '''

import sys
sys.path.append('libraries')

import pandas as pd
import joblib

from champion_weight_matrix import create_weight_value
import random

# loading the model
model = joblib.load('models/model_v1.pkl')
# print("Model loaded successfully")

# processing the input

def process_input(input_data):
    weight_vector = create_weight_value(input_data)
    categories = pd.DataFrame([input_data], columns=['T1C1', 'T1C2', 'T1C3', 'T1C4', 'T1C5', 'T2C1', 'T2C2', 'T2C3', 'T2C4', 'T2C5'])
    weights = pd.DataFrame([weight_vector], columns=['W1', 'W2', 'W3', 'W4', 'W5'])
    features = pd.concat([categories, weights], axis=1)
    return features

# making predictions

def model_predict(input_data):
    data = process_input(input_data)
    predictions = model.predict(data)
    return predictions

# calculating win percentage

def calculate_percentage(p):
    print(f"Predicted win percentage for team 1: {float((p + 1) / 2 * 100):.2f}%")
    if p >= 0:
        print("Predicted winner: team 1")
    else:
        print("Predicted winner: team 2")


# testing the model for binary accuracy

# def test():
#     with open('data/csv_files/cleaned_output.csv', 'r') as file:
#         rows = file.readlines()
#         random.shuffle(rows)
#         all = 0
#         passed = 0
#         for row in rows[:100]:
#             try:
#                 input_data = row.strip().split(',')[:10]
#                 predictions = model_predict(input_data)
#                 if (predictions[0] >= 0 and row.strip().split(',')[-1] == '100') or (predictions[0] < 0 and row.strip().split(',')[-1] == '200'):
#                     passed += 1
#                 all += 1
#             except:
#                 continue
#         print(f"Passed Tests: {passed}/{all}")
#         return passed / all * 100

''' custom input '''

try:
    input_data = input().split(',')
    # example: "Irelia,Darius,Xerath,Anivia,Sejuani,Diana,Jayce,Maokai,Neeko,Kaisa"
    predictions = model_predict(input_data)
    calculate_percentage(predictions[0])

    # sum = 0
    # for i in range(10):
    #     sum += test()
    # print(f"Average accuracy: {sum / 10:.2f}%")

except Exception as e:
    print(f"An error occurred: {e}")