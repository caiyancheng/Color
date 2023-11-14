import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from Base_Rafal import scale_inter_intra
# from Base_Rafal_Bounds import scale_inter_intra

with open('../Subject_Score.json', 'r') as fp:
    Subject_Data = json.load(fp)

Observer_ID_list = []
Condition_ID_list = []
Score_list = []
csv_data_write = {}
csv_data_write["Condition_ID"] = []
csv_data_write["MLE_Mean_Score"] = []
csv_data_write["MLE_STD_Score"] = []

for data_index in range(len(Subject_Data)):
    if Subject_Data[data_index]['Color_index'] == 2 or Subject_Data[data_index]['Color_index'] == 5:
        if Subject_Data[data_index]['display_index_1'] == 4 or Subject_Data[data_index]['display_index_2'] == 4:
            # print('Score:', Subject_Data[data_index]['score'])
            continue # Color 2 and 5 are not displayed on Display 4
    Observer_ID_list.append(Subject_Data[data_index]['Observer'])
    Condition_ID_list.append(f"D1_{Subject_Data[data_index]['display_index_1']}-"
                             f"D2_{Subject_Data[data_index]['display_index_2']}-"
                             f"C_{Subject_Data[data_index]['Color_index']}")
    Score_list.append(Subject_Data[data_index]['score'])

phi_rec, delta_rec, v_rec, w_rec, OBSs, CONDs = scale_inter_intra(observer_id_list=Observer_ID_list,
                                                                  condition_id_list=Condition_ID_list,
                                                                  quality_list=Score_list,
                                                                  no_delta=True)
# phi - 真实质量
# delta - 偏差（每个人的）
# v_rec - 不一致（每个人内部的）
# w_rec - 不一致（每个人之间的） #（会不会和delta有重复？因此取消delta）

Observer_ID_array = np.array(Observer_ID_list)
Condition_ID_array = np.array(Condition_ID_list)
Score_array = np.array(Score_list)

Mean_Score = np.zeros(CONDs.shape)
Std_Score = np.zeros(CONDs.shape)
for cond_i in range(CONDs.shape[0]):
    Mean_Score[cond_i] = np.mean(Score_array[Condition_ID_array == CONDs[cond_i]])
    Std_Score[cond_i] = np.std(Score_array[Condition_ID_array == CONDs[cond_i]])
    csv_data_write["Condition_ID"].append(CONDs[cond_i])
    csv_data_write["MLE_Mean_Score"].append(phi_rec[cond_i])
    csv_data_write["MLE_STD_Score"].append(w_rec[cond_i])

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot([0, 6], [0, 6], '--k')
plt.scatter(Mean_Score, phi_rec)
plt.xlabel('Mean Score')
plt.ylabel('MLE Mean Score')

plt.subplot(1, 2, 2)
plt.plot([0, 1.5], [0, 1.5], '--k')
plt.scatter(Std_Score, w_rec)
plt.xlabel('Std Score')
plt.ylabel('MLE Std Score')

plt.tight_layout()
plt.show()

df = pd.DataFrame(csv_data_write)
df.to_csv('MLE_CSV.csv', index=False)