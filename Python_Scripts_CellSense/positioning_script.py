import numpy as np
import scipy
import numpy.matlib as mtlb
from scipy.optimize import root, minimize, least_squares
import matplotlib.pyplot as plt


# Function to extract the SRS delays from the csv files
def str2data(srs_delay_str, K, srs_delay):
    delay = ''
    n = 0
    k = 0
    for i in range(0, len(srs_delay_str)):
        if(k == K):
            return
        
        if(srs_delay_str[i] == ','):
            srs_delay[k][n] = int(delay)
            delay = ''
            n = n+1
            continue
        
        if(srs_delay_str[i] == '\n'):
            srs_delay[k][n] = int(delay)
            delay = ''
            k = k+1
            n = 0
            continue

        delay = delay + srs_delay_str[i]


# Function to model the pseudorange difference and estimate the UE location
def func(var, x, y, z, delta_meas_delay):
    x_ue, y_ue, z_ue = var
    # Example residuals
    r = 0
    for i in range(1, 4):
        r += (np.sqrt((x[i]-x_ue)**2 + (y[i]-y_ue)**2 + (z[i]-z_ue)**2) - c*delta_meas_delay[i-1] - np.sqrt((x[0]-x_ue)**2 + (y[0]-y_ue)**2 + (z[0]-z_ue)**2))**2
        
    return r


# Function to perform calibration using the SRS delays and known location of the UE and APs
def cal_func(x, y, z, N_ant, cal_delay, ue_cal, c):
    cal_SDO = np.zeros([np.shape(cal_delay)[0], N_ant-1])
    x_ue = ue_cal[0]
    y_ue = ue_cal[1]
    z_ue = ue_cal[2]

    for iter in range(0, np.shape(cal_delay)[0]):
        cal_SDO[iter, 0] = cal_delay[iter, 1] - cal_delay[iter, 0] - (np.sqrt((x[1]-x_ue)**2 + (y[1]-y_ue)**2 + (z[1]-z_ue)**2) - \
                                                                      np.sqrt((x[0]-x_ue)**2 + (y[0]-y_ue)**2 + (z[0]-z_ue)**2))/c*1e9
        cal_SDO[iter, 1] = cal_delay[iter, 2] - cal_delay[iter, 0] - (np.sqrt((x[2]-x_ue)**2 + (y[2]-y_ue)**2 + (z[2]-z_ue)**2) - \
                                                                      np.sqrt((x[0]-x_ue)**2 + (y[0]-y_ue)**2 + (z[0]-z_ue)**2))/c*1e9
        cal_SDO[iter, 2] = cal_delay[iter, 3] - cal_delay[iter, 0] - (np.sqrt((x[3]-x_ue)**2 + (y[3]-y_ue)**2 + (z[3]-z_ue)**2) - \
                                                                      np.sqrt((x[0]-x_ue)**2 + (y[0]-y_ue)**2 + (z[0]-z_ue)**2))/c*1e9

    return cal_SDO


# Constants for the experiment (True UE locations, APs locations, no. of APs, and speed of light)
x = np.array([-0.1, 2.45, 6.6, 0])
y = np.array([1.8, 4.5, 0.5, 4.8])
z = np.array([0, 0.1, 0.1, 0.15])
N_ant = 4
c = 299792458
ue = np.array([[2.9, 1.94, -0.2], [4.95, 3.05, -0.2]])
ue_cal = np.array([1.2, 3.06, -0.2])

# Constants for the running instance
K_cal = 100
K_ue = 100
win = 10

# Collecting the string data from the .csv files
with open("/home/bkumar2/Documents/Research/O-RAN positioning/Data/OutdoorLoc1/srs_delay", 'r') as f:
    cal_delay_str = f.read()
f.close()

with open("/home/bkumar2/Documents/Research/O-RAN positioning/Data/OutdoorLoc2/srs_delay", 'r') as f:
    ue1_delay_str = f.read()
f.close()

with open("/home/bkumar2/Documents/Research/O-RAN positioning/Data/OutdoorLoc3/srs_delay", 'r') as f:
    ue2_delay_str = f.read()
f.close()

# with open("/home/bkumar2/Documents/Research/O-RAN positioning/Data/IndoorLoc3/srs_delay", 'r') as f:
#     ue3_delay_str = f.read()
# f.close()

cal_delay = np.zeros([K_cal, N_ant])
ue_delay = np.zeros([K_ue, N_ant, 2])

# Convert the string data into SRS delays
str2data(cal_delay_str, K_cal, cal_delay)
str2data(ue1_delay_str, K_ue, ue_delay[:, :, 0])
str2data(ue2_delay_str, K_ue, ue_delay[:, :, 1])
# str2data(ue3_delay_str, K_ue, ue_delay[:, :, 2])

# Calculate the no. and indices of outliers in the collected SRS delays based on if the difference between SRS delays is too big or if the individual values are deviant
outliers = np.array(np.abs(cal_delay[:, 1] - cal_delay[:, 0]) > 100, dtype=int) + np.array(np.abs(cal_delay[:, 2] - cal_delay[:, 0]) > 100, dtype=int) + \
            np.array(np.abs(cal_delay[:, 3] - cal_delay[:, 0]) > 100, dtype=int) + np.array(np.abs(cal_delay[:, 0]) > 1000, dtype=int)
idx_outliers = [arg for arg in range(0, len(outliers)) if outliers[arg] != 0]
cal_indices = np.setdiff1d(np.arange(0, K_cal, 1), idx_outliers)

cal_SDO = cal_func(x, y, z, N_ant, cal_delay[cal_indices, :], ue_cal, c)

# cal_SDO = cal_delay[cal_indices, 1:4] - np.tile(cal_delay[cal_indices, 0:1], [1, 3])

plt.figure(3)
plt.plot(np.arange(0, len(cal_indices), 1), cal_SDO, linewidth=2)
plt.xlabel("SRS measurement snapshots", fontsize=24)
plt.ylabel("Delay in ns", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(['Diff. between Ant 1 and 0', 'Diff. between Ant 2 and 0', 'Diff. between Ant 3 and 0'], fontsize=30)

mean_cal_SDO = np.average(cal_SDO, axis=0)

for iter0 in range(0, np.shape(ue_delay)[2]):
    # Calculate the no. and indices of outliers in the collected SRS delays based on if the difference between SRS delays is too big or if the individual values are deviant
    outliers = np.array(np.abs(ue_delay[:, 1, iter0] - ue_delay[:, 0, iter0]) > 100, dtype=int) + np.array(np.abs(ue_delay[:, 2, iter0] - ue_delay[:, 0, iter0]) > 100, dtype=int) + \
            np.array(np.abs(ue_delay[:, 3, iter0] - ue_delay[:, 0, iter0]) > 100, dtype=int) + np.array(np.abs(ue_delay[:, 0, iter0]) > 1000, dtype=int)
    idx_outliers = [arg for arg in range(0, len(outliers)) if outliers[arg] != 0]

    no_err_indices = np.setdiff1d(np.arange(0, K_ue, 1), idx_outliers)

    delta_meas_delay = ue_delay[no_err_indices, 1:4, iter0] - np.tile(ue_delay[no_err_indices, 0:1, iter0], [1,3])

    plt.figure(iter0+4)
    plt.plot(np.arange(0, len(no_err_indices), 1), delta_meas_delay, linewidth=2)
    plt.xlabel("SRS measurement snapshots", fontsize=24)
    plt.ylabel("Delay in ns", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(['Diff. between Ant 1 and 0', 'Diff. between Ant 2 and 0', 'Diff. between Ant 3 and 0'], fontsize=30)

    plt.figure(iter0)
    plt.scatter(x, y, marker='*', linewidth=15)
    plt.scatter(ue[iter0, 0], ue[iter0, 1], marker='x', linewidth=15)

    est_ue = np.zeros([len(no_err_indices) // win, 3])

    # Estimate the location of UE by averaging the SRS delays across a window length of 'win'
    for iter1 in range(0, len(no_err_indices) // win):
        mean_meas_delay = np.average(delta_meas_delay[iter1*win:(iter1+1)*win, :], axis=0)
        # mean_meas_delay = mean_meas_delay - mean_cal_SDO
        mean_meas_delay = 1e-9*mean_meas_delay

        res = least_squares(func, [0,0,0], args=(x, y, z, mean_meas_delay))
        est_ue[iter1, :] = res.x

    plt.scatter(est_ue[:, 0], est_ue[:, 1], color= 'r' , marker='o', linewidth=10)
    plt.xlabel("X-axis (meters)", fontsize=24)
    plt.ylabel("Y-axis (meters)", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(["BS AP locations", "True UE location", "Estimated UE locations"], fontsize=30)

plt.show()
# print("The position of the UE is [{}, {}, {}] \n".format(ue_pos[0], ue_pos[1], ue_pos[2]))
