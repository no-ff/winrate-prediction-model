''' importing libraries '''

import sys
sys.path.append('libraries')

import csv
import json
import pandas as pd
import numpy as np

''' main code '''

def create_weight_value(row_data):
  # champion pairing dictionary
  CHDICT = json.load(open("data/champ_dict.json"))
  
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
  with open("data/csv_files/cleaned_output.csv", 'r') as data_file:
    next(data_file)
    csv_reader = csv.reader(data_file)
  
    weight_matrix = []
    for line in csv_reader:
      weight_vector = create_weight_value(line)
      line[10:10] = weight_vector
      weight_matrix.append(line)
      
    with open("data/csv_files/cleaned_output.csv", 'w', newline='') as data_file:
      csv_writer = csv.writer(data_file)
      csv_writer.writerow(['T1C1','T1C2','T1C3','T1C4','T1C5','T2C1','T2C2','T2C3','T2C4','T2C5','W1','W2','W3','W4','W5','GD','WL'])
      csv_writer.writerows(weight_matrix)
  
  return weight_matrix

''' Converting weight matrix into pandas dataframe '''   

# matrix = np.array(create_weight_matrix())
# dataframe = pd.DataFrame(matrix)
# print(dataframe) 

create_weight_matrix()