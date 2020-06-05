import json
import os
import numpy as np
import pickle
import sys
import time
import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

import os

import re


if __name__ == '__main__':
    THE_FOLDER = "./json"

    json_str = '{\n'

    for file in sorted(os.listdir(THE_FOLDER)):
        with open(f"./json/{file}", 'r') as f:
            data = f.read()
        end = file.find('.json')
        id = file[6:end]
        json_str += "\t\"" + str(id) + "\":"
        json_str += data + ",\n" 
    
    #to delete trailing comma
    json_str = json_str[0:-2]
    json_str += '\n}'

    f = open('submit.json', 'w')
    f.write(json_str)