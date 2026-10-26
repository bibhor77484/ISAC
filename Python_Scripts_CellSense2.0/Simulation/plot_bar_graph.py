import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

barWidth = 0.2
fig = plt.subplots(figsize =(12, 8)) 

## Old values
# values_cellsense = [0.74, 0.94, 1] # Detection probability
# values_cellsense = [0.10, 0.12, 0.02] # False alarm probability
# values_cellsense = [1.43, 0.33, 0.28] # RMSE
# values_cellsense = [1.19, 0.27, 0.25] # Mean localization error (meters)
# std_cellsense = [0.80, 0.19, 0.13] # Standard deviation of localization error (meters)

## New values
values_cellsense = [1.69, 0.66, 0.56] # Mean localization error (meters)
sd_cellsense = [1.03, 0.54, 0.44] # Standard deviation of localization error (meters)
# values_cellsense = [0.75, 0.95, 1] # Detection probability
# values_cellsense = [0.15, 0.16, 0.01] # False alarm probability

# values_cellsense2 = [1.4, 0.53, 0.52] # RMSE
# values_cellsense2 = [0.76, 0.95, 1] # Detection probability
# values_cellsense2 = [0.14, 0.13, 0.03] # False alarm probability


# values_2dfft = [0.22, 0.84, 0.88] # Detection probability
# values_2dfft = [0.88, 0.42, 0.00] # False alarm probability
# values_2dfft = [3.43, 2.38, 1.65] # RMSE
values_2dfft = [2.99, 2.07, 1.34] # Mean localization error (meters)
sd_2dfft = [1.46, 1.12, 0.98] # Standard deviation of localization error (meters)

br1 = np.arange(len(values_cellsense)) 
br2 = [x + 1.5*barWidth for x in br1] 

plt.bar(br1, values_cellsense, yerr=sd_cellsense, capsize=10,
        error_kw={'elinewidth':3, 'ecolor':'black'}, color='b', width=barWidth,
        edgecolor='grey', label='CellSense')
plt.bar(br2, values_2dfft, yerr=sd_2dfft, capsize=10,
        error_kw={'elinewidth':3, 'ecolor':'black'}, color='r', width=barWidth,
        edgecolor='grey', label='2D-FFT')

plt.xlabel('Type of environment', fontweight ='bold', fontsize = 75) 
plt.ylabel('Mean error ± SD (meters)', fontweight ='bold', fontsize = 75) 
plt.xticks([r + 0.75*barWidth for r in range(len(values_cellsense))], 
        ['Indoor', 'Outdoor', 'Open'], fontsize=75)
plt.yticks(fontsize=75)

plt.legend(fontsize=75)
plt.show() 

# def est_pos(d, l_hat, theta_hat):
#     x1 = (d + l_hat)*np.sin(theta_hat)
#     y1 = (d + l_hat)*np.cos(theta_hat)
#     m = (y1 - d)/x1

#     x_mid = x1/2
#     y_mid = (y1 + d)/2

#     x_s_hat = (x_mid/m + y_mid)/(1/m + 1/np.tan(theta_hat))
#     y_s_hat = x_s_hat/np.tan(theta_hat)

#     return x_s_hat, y_s_hat


# D = 3.6 # distance of the UE from the gNB (in m)
# c = 299792458

# x_est, y_est = est_pos(3.6, 4.07*c/1e9, 53.08*np.pi/180)

# print(x_est, y_est)
# print(np.sqrt((x_est - 1.6)**2 + (y_est - 1.8)**2))