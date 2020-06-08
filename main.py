import numpy as np
import os
import json
from solver import *
from eva import evaluate
import re

def post_process(ans_list, ans_idx):
    f = open('submit.json', 'w')
    json_str = '{\n'
    for i in range(len(ans_list)):
        title = "\t\"" + str(ans_idx[i]) + "\":"
        block = json.dumps(ans_list[i], separators=(',',':'), indent=4)
        json_str += (title + block)
        if i != len(ans_list)-1:
            json_str += ",\n"
    json_str += '\n}'
    f.write(json_str)
    f.close()

if __name__ == '__main__':
    THE_FOLDER = "./AIcup_testset_ok"

    ans_list = []
    ans_idx = []

    '''
    # for testing 
    avg_score = []
    '''
    
    for the_dir in os.listdir(THE_FOLDER):
        
        print(the_dir)
        if not os.path.isdir(THE_FOLDER + "/" + the_dir):
            continue
        json_path = THE_FOLDER + "/" + the_dir + f"/{the_dir}_vocal.json"
        
        '''
        # this part is for testing
        gt_path = THE_FOLDER + "/" + the_dir + "/" + the_dir + "_groundtruth.txt"
        gtdata = np.loadtxt(gt_path)
        '''

        with open(json_path, 'r') as json_file:
            data = json.loads(json_file.read())
        ans = process(data)
        
        # print(len(ans))
        ans_list.append(ans)
        ans_idx += [the_dir]

        '''
        score = evaluate(np.array(ans), np.array(gtdata))
        print(f'score: {score}')
        avg_score += [score]
        '''
    '''
    print(f'avg_score: {np.mean(np.array(avg_score))}')
    '''

    # output the result
    post_process(ans_list, ans_idx)
