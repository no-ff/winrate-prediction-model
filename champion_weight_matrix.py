import csv
import json
import pandas as pd
import numpy as np

def create_weight_value(row_data):
  # champion pairing dictionary
  CHDICT = json.load(open("json_files/champ_dict.json"))
  
  # constants
  TEAM_1 = row_data[:5]
  TEAM_2 = row_data[5:10]
  POSITIONS = ['top','jungle','mid','adc','support']
  
  # the average weight of all the champion weights
  weight_vector = []

  # compare each team 1 champion to the corresponding team 2 champion
  for i in range(len(TEAM_1)):
    # checks if a path in the json can be made with the champion
    if TEAM_1[i] in CHDICT.keys() and POSITIONS[i] in CHDICT[TEAM_1[i]].keys() \
    and TEAM_2[i] in CHDICT[TEAM_1[i]][POSITIONS[i]].keys():
      weight_vector.append(float(CHDICT[TEAM_1[i]][POSITIONS[i]][TEAM_2[i]]))
    else:
      weight_vector.append(50)

  return weight_vector

def create_weight_matrix():
  with open("csv_files/cleaned_output.csv", 'r') as data_file:
    next(data_file)
    csv_reader = csv.reader(data_file)
  
    weight_matrix = []
    for line in csv_reader:
      weight_matrix.append(create_weight_value(line))
  
  return weight_matrix

''' Converting weight matrix into pandas dataframe

matrix = np.array(create_weight_matrix())
dataframe = pd.DataFrame(matrix)
print(dataframe)
'''    
