import json
import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm

json_data_list = []
Subject_score_data_path = r'E:\About_Cambridge\All Research Projects\Color\ObserverMetamerism\ObserverMetamerism\Data\ObsResponses'

display_pattern = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
# Colors 2 and 5 are not evaluated on display 4
subject_data_dir = os.listdir(Subject_score_data_path)

for subject_file in tqdm(subject_data_dir):
    if subject_file.endswith('test.csv'):
        print('Exclude Test')
    elif subject_file.endswith('.csv'):
        subject_data = pd.read_csv(os.path.join(Subject_score_data_path,subject_file), header=None)
        for row_index in range(6):
            display_index_1, display_index_2 = display_pattern[row_index]
            observer_name = subject_file.split('.')[0].split('_')[-1]
            for column_index in range(22):
                color_index = column_index % 11 + 1# 1 ~ 11
                score = int(subject_data[color_index - 1][row_index])
                if score == 0:
                    X = 1
                json_data_dict = {'Observer':observer_name,
                                  'Color_index':color_index,
                                  'display_index_1':display_index_1,
                                  'display_index_2':display_index_2,
                                  'score': score}
                json_data_list.append(json_data_dict)

with open('E:/Py_codes/Color/Subject_Score.json', 'w') as fp:
    json.dump(json_data_list, fp)




