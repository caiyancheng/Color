import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
import sys
import os
sys.path.append('E:\Py_codes/ColorVideoVDP')
from pycvvdp.dolby_ictcp import ictcp
import torch
import json
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr


# Read data from CSV files
all_data = pd.read_csv('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/YC_AllData.csv')
# all_data.drop(all_data.index[4620:4751], inplace=True) # Test不包含
ciexyz31_1 = pd.read_csv('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/AuxData/ciexyz31_1.csv', header=None).to_numpy()

obs_1000_cmf_struct = loadmat("E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/AuxData/IndividualObs_2deg.mat")
Obs_1000_CMFs = obs_1000_cmf_struct['xyz_CMFs']
original_wavelengths = np.arange(390, 785, 5)
desired_wavelengths = np.arange(380, 781, 1)
Obs_1000_CMFs_1nm = np.zeros((len(desired_wavelengths), 3, 1000))
for i in range(1000):
    for j in range(3):
        f = interp1d(original_wavelengths, Obs_1000_CMFs[:, j, i], kind='linear', bounds_error=False, fill_value="extrapolate")
        Obs_1000_CMFs_1nm[:, j, i] = f(desired_wavelengths)

display_1_spd = pd.read_csv('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/Spectra/C2_Spectra.csv', header=None).to_numpy()
display_2_spd = pd.read_csv('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/Spectra/X310_Spectra.csv', header=None).to_numpy()
display_3_spd = pd.read_csv('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/Spectra/Projector_Spectra.csv', header=None).to_numpy()
display_4_spd = pd.read_csv('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/Spectra/VG246_Spectra.csv', header=None).to_numpy()
display_spds = [display_1_spd, display_2_spd, display_3_spd, display_4_spd]

exp_num = len(all_data)
color_rgb_s = pd.read_csv("E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/all_11_colors_rgb.csv", header=None).to_numpy() / 256.0
display_patterns = np.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
display_configurations = np.array([[-1, 1, 2, 3], [-1, -1, 4, 5], [-1, -1, -1, 6], [-1, -1, -1, -1]])
num_obs = Obs_1000_CMFs.shape[2]
e_cell = np.zeros((6, 11), dtype=object)
score_cell = np.empty((6, 11), dtype=object)

deltaE_ITP_class = ictcp()
# Calculate E_Cell
# for display_pattern_index in range(6):
#     display_1 = display_patterns[display_pattern_index, 0]
#     display_2 = display_patterns[display_pattern_index, 1]
#     for color_index in range(11):
#         display_spd_1 = display_spds[display_1 - 1]
#         display_spd_2 = display_spds[display_2 - 1]
#         E_set = []
#         for obs in range(num_obs):
#             X1, X2, Y1, Y2, Z1, Z2 = 0, 0, 0, 0, 0, 0
#             for lamda in range(380,781,1):
#                 X1 = X1 + display_spd_1[color_index, lamda - 380] * Obs_1000_CMFs_1nm[lamda - 380, 0, obs]
#                 X2 = X2 + display_spd_2[color_index, lamda - 380] * Obs_1000_CMFs_1nm[lamda - 380, 0, obs]
#                 Y1 = Y1 + display_spd_1[color_index, lamda - 380] * Obs_1000_CMFs_1nm[lamda - 380, 1, obs]
#                 Y2 = Y2 + display_spd_2[color_index, lamda - 380] * Obs_1000_CMFs_1nm[lamda - 380, 1, obs]
#                 Z1 = Z1 + display_spd_1[color_index, lamda - 380] * Obs_1000_CMFs_1nm[lamda - 380, 2, obs]
#                 Z2 = Z2 + display_spd_2[color_index, lamda - 380] * Obs_1000_CMFs_1nm[lamda - 380, 2, obs]
#             XYZ1 = torch.tensor([X1*683, Y1*683, Z1*683])[None,:,None,None,None]
#             XYZ2 = torch.tensor([X2*683, Y2*683, Z2*683])[None,:,None,None,None]
#             delta_E_ITP = deltaE_ITP_class.predict_xyz(XYZ1, XYZ2)
#             E_set.append(delta_E_ITP)
#         e_cell[display_pattern_index, color_index] = E_set

with open(r'E:\Py_codes\Color/e_cell.json', 'r') as fp:
    data = json.load(fp)
    e_cell = np.array(data)

# with open(r'E:\Py_codes\Color/e_cell.json', 'w') as fp:
#     json.dump(e_cell.tolist(), fp=fp)

# Initialize variables
for i in range(6):
    for j in range(11):
        score_cell[i, j] = []
# Populate score_Cell
for i in range(exp_num):
    display_1 = int(all_data.loc[i, "Display1"])
    display_2 = int(all_data.loc[i, "Display2"])
    color_index = int(all_data.loc[i, "ColorIndex"])
    display_pattern_index = display_configurations[display_1 - 1, display_2 - 1]
    score = all_data.loc[i, "MatchingScore"]
    score_cell[display_pattern_index - 1, color_index - 1].append(score)

# Calculate means and std
e_means = np.zeros((6, 11))
e_std = np.zeros((6, 11))
score_means = np.zeros((6, 11))
score_std = np.zeros((6, 11))
point_color = np.zeros((6, 11, 3))
point_display_pattern = np.zeros((6, 11))

for i in range(6):
    for j in range(11):
        if all(score == 0 for score in score_cell[i, j]):
            e_means[i, j] = np.nan
            e_std[i, j] = np.nan
            score_means[i, j] = np.nan
            score_std[i, j] = np.nan
        else:
            e_means[i, j] = np.mean(e_cell[i, j])
            e_std[i, j] = np.std(e_cell[i, j])
            score_means[i, j] = np.mean(score_cell[i, j])
            score_std[i, j] = np.std(score_cell[i, j])

        point_color[i, j, :] = color_rgb_s[j, :]
        point_display_pattern[i, j] = i

# Flatten the arrays for plotting
e_means_flat = e_means.flatten()
e_std_flat = e_std.flatten()
score_means_flat = score_means.flatten()
score_std_flat = score_std.flatten()
point_color_flat = point_color.reshape((-1, 3))
point_display_pattern_flat = point_display_pattern.flatten()

valid_indices = ~np.isnan(e_means_flat) & ~np.isnan(score_means_flat)

# Plot the scatter plot with error bars
plt.figure(figsize=(10,5))
plt.scatter(e_std_flat[valid_indices], score_std_flat[valid_indices], c=point_color_flat[valid_indices], edgecolors=point_color_flat[valid_indices])
for indice in range(valid_indices.size):
    if not valid_indices[indice]:
        continue
    # elif e_means_flat[indice] > 5:
    #     continue
    plt.text(e_std_flat[indice], score_std_flat[indice] - 0.02,
             "{}, {}".format(display_patterns[int(point_display_pattern_flat[indice]), 0],
                             display_patterns[int(point_display_pattern_flat[indice]), 1]),
             horizontalalignment='center')

correlation_coefficient_Pearson = pearsonr(e_std_flat[valid_indices], score_std_flat[valid_indices]).correlation
correlation_coefficient_Kendall = kendalltau(e_std_flat[valid_indices], score_std_flat[valid_indices]).correlation
correlation_coefficient_Spearman = spearmanr(e_std_flat[valid_indices], score_std_flat[valid_indices]).correlation

X_axis_begin = 8
Y_axis_begin = 1.0
Y_axis_gap = 0.03

for i in range(11):
    y = - (i - 9) * Y_axis_gap + Y_axis_begin
    plt.scatter(X_axis_begin + 1, y, s=60, c=[color_rgb_s[i, :]], edgecolors=[color_rgb_s[i, :]])
    plt.text(X_axis_begin, y, 'Color {}'.format(i + 1), horizontalalignment='left', verticalalignment='center')

plt.text(X_axis_begin, Y_axis_begin - 3 * Y_axis_gap, 'Display 1 - LG C2', horizontalalignment='left', verticalalignment='center')
plt.text(X_axis_begin, Y_axis_begin - 4 * Y_axis_gap, 'Display 2 - Sony X310', horizontalalignment='left', verticalalignment='center')
plt.text(X_axis_begin, Y_axis_begin - 5 * Y_axis_gap, 'Display 3 - Samsung Laser Projector', horizontalalignment='left', verticalalignment='center')
plt.text(X_axis_begin, Y_axis_begin - 6 * Y_axis_gap, 'Display 4 - ASUS VG246', horizontalalignment='left', verticalalignment='center')
plt.text(X_axis_begin, Y_axis_begin - 7 * Y_axis_gap, f'Pearson Correlation: {correlation_coefficient_Pearson}', horizontalalignment='left', verticalalignment='center')
plt.text(X_axis_begin, Y_axis_begin - 8 * Y_axis_gap, f'Kendall Correlation: {correlation_coefficient_Kendall}', horizontalalignment='left', verticalalignment='center')
plt.text(X_axis_begin, Y_axis_begin - 9 * Y_axis_gap, f'Spearman Correlation: {correlation_coefficient_Spearman}', horizontalalignment='left', verticalalignment='center')

plt.xlim([0, 9])
plt.xlabel('deltaE 2000 (Standard Deviation of 1000 values)')
plt.ylabel('Score (Standard Deviation of 74 values)')
plt.title('74 Subject Scores and 1000 CMFs deltaE 2000 Standard deviation points')
plt.savefig('E:/About_Cambridge/All Research Projects/Color/ObserverMetamerism/ObserverMetamerism/Data/deltaE_subjective_pure_std_no_zero.png')
plt.show()
