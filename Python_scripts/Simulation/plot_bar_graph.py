import numpy as np
import matplotlib.pyplot as plt
import scipy

barWidth = 0.2
fig = plt.subplots(figsize =(12, 8)) 

# values_cellsense = [0.74, 0.94, 1] # Detection probability
# values_cellsense = [0.10, 0.12, 0.02] # False alarm probability
values_cellsense = [1.43, 0.33, 0.28] # RMSE

# values_2dfft = [0.22, 0.84, 0.88] # Detection probability
# values_2dfft = [0.88, 0.42, 0.00] # False alarm probability
values_2dfft = [3.43, 2.38, 1.65] # RMSE

br1 = np.arange(len(values_cellsense)) 
br2 = [x + 1.5*barWidth for x in br1] 

plt.bar(br1, values_cellsense, color ='b', width = barWidth, 
        edgecolor ='grey', label ='CellSense') 
plt.bar(br2, values_2dfft, color ='r', width = barWidth, 
        edgecolor ='grey', label ='2D-FFT') 

plt.xlabel('Type of environment', fontweight ='bold', fontsize = 50) 
plt.ylabel('RMSE (meters)', fontweight ='bold', fontsize = 50) 
plt.xticks([r + 0.75*barWidth for r in range(len(values_cellsense))], 
        ['Indoor', 'Outdoor', 'Open'], fontsize=50)
plt.yticks(fontsize=50)

plt.legend(fontsize=50)
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