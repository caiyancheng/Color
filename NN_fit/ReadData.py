import pandas as pd
import torch

def Read_Data(path='../write_matlab_results_data.csv', pattern='score'):
    data_all = pd.read_csv(path)
    input_list = []
    target_list = []
    for row in range(len(data_all)):
        E_mean = data_all['E_mean'][row]
        E_std = data_all['E_std'][row]
        MLE_std = data_all['MLE_std'][row]
        Score_std = data_all['Score_std'][row]
        input_list.append(torch.tensor([E_mean, E_std]))
        if pattern == 'score':
            target_list.append(Score_std)
        elif pattern == 'MLE':
            target_list.append(MLE_std)
        else:
            raise KeyError('Target Pattern is Wrong')
    input_tensor = torch.stack(input_list, 0)
    target_tensor = torch.tensor(target_list)
    return input_tensor, target_tensor

if __name__ == '__main__':
    Read_Data()


