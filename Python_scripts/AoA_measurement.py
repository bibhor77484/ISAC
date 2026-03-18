import numpy as np
import scipy
import numpy.matlib as mtlb
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
import matplotlib


def str2data(chan_str, N, K, chan):
    channel = ''
    n = 0
    k = 0
    for i in range(0, len(chan_str)):
        if (n == N-1) and (k == K-1):
            return
        
        if(chan_str[i] == '\n'):
            for l in range(0, len(channel)):
                if channel[l] == ',':
                    real = channel[0:l]
                    imag = channel[l+1:len(channel)]

            chan[k][n][0] = int(real)
            chan[k][n][1] = int(imag)
            channel = ''
            n = n+1
            if n >= N:
                n = 0
                k = k+1
            continue

        channel = channel + chan_str[i]


with open("/home/bkumar2/Documents/Research/ISAC/Datasets/Same_location_1_session/chanF_ant0", 'r') as f0:
    chanF0_str = f0.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/Same_location_1_session/chanF_ant1", 'r') as f1:
    chanF1_str = f1.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/Same_location_1_session/chanF_ant2", 'r') as f2:
    chanF2_str = f2.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/Same_location_1_session/chanF_ant3", 'r') as f3:
    chanF3_str = f3.read()

N_chanF = 1024

# Edit these parameters based on the dataset collected
K = 122 # Total no. of times data is collected
idx_calibration = 50 # Index until which data is used for calibration
idx_est1 = 60 # Starting index for estimation
idx_est2 = 110 # Ending index for estimation
win_size = 10 # Window size for averaging 

chanF0 = np.zeros([K, N_chanF, 2])
chanF1 = np.zeros([K, N_chanF, 2])
chanF2 = np.zeros([K, N_chanF, 2])
chanF3 = np.zeros([K, N_chanF, 2])

str2data(chanF0_str, N_chanF, K, chanF0)
str2data(chanF1_str, N_chanF, K, chanF1)
str2data(chanF2_str, N_chanF, K, chanF2)
str2data(chanF3_str, N_chanF, K, chanF3)

complex_chanF0 = chanF0[:, :, 0] + 1j*chanF0[:, :, 1]
complex_chanF1 = chanF1[:, :, 0] + 1j*chanF1[:, :, 1]
complex_chanF2 = chanF2[:, :, 0] + 1j*chanF2[:, :, 1]
complex_chanF3 = chanF3[:, :, 0] + 1j*chanF3[:, :, 1]

complex_chanF0 = np.concatenate((complex_chanF0[:, 718:1024], complex_chanF0[:, 0:270]), axis=1)
complex_chanF1 = np.concatenate((complex_chanF1[:, 718:1024], complex_chanF1[:, 0:270]), axis=1)
complex_chanF2 = np.concatenate((complex_chanF2[:, 718:1024], complex_chanF2[:, 0:270]), axis=1)
complex_chanF3 = np.concatenate((complex_chanF3[:, 718:1024], complex_chanF3[:, 0:270]), axis=1)

offset_chanF0 = complex_chanF0*np.conjugate(complex_chanF0)
offset_chanF1 = complex_chanF1*np.conjugate(complex_chanF0)
offset_chanF2 = complex_chanF2*np.conjugate(complex_chanF0)
offset_chanF3 = complex_chanF3*np.conjugate(complex_chanF0)

cal_mean_chanF0 = np.average(offset_chanF0[0:idx_calibration, :], axis=0)
cal_mean_chanF1 = np.average(offset_chanF1[0:idx_calibration, :], axis=0)
cal_mean_chanF2 = np.average(offset_chanF2[0:idx_calibration, :], axis=0)
cal_mean_chanF3 = np.average(offset_chanF3[0:idx_calibration, :], axis=0)

cal_mean_arr = np.array([cal_mean_chanF0, cal_mean_chanF1, cal_mean_chanF2, cal_mean_chanF3])
cal_mean_arr = np.average(cal_mean_arr, axis=1)

cal_phases = np.angle(cal_mean_arr)

# Antenna spacing from antenna 0 in wavelengths
d = np.array([0, 0.5, 1, 1.5])

for idx_win in range(0, int((idx_est2 - idx_est1)/win_size)):
    est_mean_chanF0 = np.average(offset_chanF0[idx_est1+idx_win*win_size:idx_est1+(idx_win+1)*win_size, :], axis=0)
    est_mean_chanF1 = np.average(offset_chanF1[idx_est1+idx_win*win_size:idx_est1+(idx_win+1)*win_size, :], axis=0)
    est_mean_chanF2 = np.average(offset_chanF2[idx_est1+idx_win*win_size:idx_est1+(idx_win+1)*win_size, :], axis=0)
    est_mean_chanF3 = np.average(offset_chanF3[idx_est1+idx_win*win_size:idx_est1+(idx_win+1)*win_size, :], axis=0)

    est_mean_arr = np.array([est_mean_chanF0, est_mean_chanF1, est_mean_chanF2, est_mean_chanF3])
    est_mean_arr = np.average(est_mean_arr, axis=1)

    est_mean_arr = est_mean_arr*np.exp(-1j*cal_phases)

    # Code for beam sweeping here and finding the strongest direction.
    beamforming = np.zeros(len(np.arange(-np.pi/2, np.pi/2, 0.1)))
    theta = np.arange(-np.pi/2, np.pi/2, 0.1)
    for i in range(0, len(theta)):
        beamforming[i] = np.abs(np.sum(est_mean_arr*np.exp(-1j*2*np.pi*d*np.sin(theta[i]))))

    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)
    plt.figure(idx_win)
    plt.polar(theta, beamforming)
    
plt.show()