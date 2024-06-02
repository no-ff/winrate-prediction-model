import sys
sys.path.append('libraries')

import json

CHDICT = json.load(open("data/champ_dict.json"))
POSITIONS = ['top','jungle','mid','adc','support']

def get_weight(c1, c2, idx1):
    if c1 in CHDICT.keys() and POSITIONS[idx1] in CHDICT[c1].keys() \
    and c2 in CHDICT[c1][POSITIONS[idx1]].keys():
      return float(CHDICT[c1][POSITIONS[idx1]][c2])
    return 50
