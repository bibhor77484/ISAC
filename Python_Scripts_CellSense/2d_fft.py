import numpy as np
import scipy
import numpy.matlib as mtlb
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
import time
import sys

time0 = time.time()

def chanstr2data(chan_str, N, K, chan):
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


def delstr2data(srs_delay_str, K, srs_delay):
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


# Function for circular convolution with symmetric h (add another line if h is not symmetric)
def circ_con(h, x, N):
    y = np.zeros(N, dtype=np.complex128)
    
    for n in range(0, N):
        y[n] = np.sum(x*np.roll(h, n))
    
    return y


# Function to find LoS path using the estimated AoA and ToF
def find_los(t_est, theta_est, alpha_est, aoa_ue):
    alpha_max = np.max(np.abs(alpha_est))
    order = np.argsort(np.abs(theta_est - aoa_ue)) # Arrange the paths so that the angles are arranged in order of closeness to the UE signal direction. We do this to avoid cases when a path whose delay is almost equivalent to the LoS gets deleted.
    t_est = t_est[order]
    theta_est = theta_est[order]
    alpha_est = alpha_est[order]
    
    while(len(t_est) > 0): # Continue until there is a path left to process
        l_ue = np.argmin(t_est) # Find the path index with the minimum transmit delay
        if (np.abs(theta_est[l_ue] - aoa_ue) > 10) or (np.abs(alpha_est[l_ue]) < 0.5*alpha_max): # Ignore the path if it is not coming from the UE or if it is too weak. LoS from the UE should be one of the strongest paths if not the strongest.
            t_est = np.delete(t_est, l_ue)
            theta_est = np.delete(theta_est, l_ue)
            alpha_est = np.delete(alpha_est, l_ue)
        else:
            break

    return t_est, theta_est, alpha_est, l_ue


### Function to estimate the position of the object of interest from the estimated range, angle of the reflected path and distance of the UE from the gNB
def est_pos(d, l_hat, theta_hat):
    x1 = (d + l_hat)*np.sin(theta_hat)
    y1 = (d + l_hat)*np.cos(theta_hat)
    m = (y1 - d)/x1

    x_mid = x1/2
    y_mid = (y1 + d)/2

    x_s_hat = (x_mid/m + y_mid)/(1/m + 1/np.tan(theta_hat))
    y_s_hat = x_s_hat/np.tan(theta_hat)

    return x_s_hat, y_s_hat


# Compute the CDF from data and x values (x values are the values at which we want to evaluate the CDF)
def cdf(err, x):
    CDF = np.zeros(len(x))
    for iter in range(0, len(x)):
        CDF[iter] = np.sum(err <= x[iter])/len(err)
    return CDF





with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/chanF_ant0", 'r') as f0:
    chanF0_str = f0.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/chanF_ant1", 'r') as f1:
    chanF1_str = f1.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/chanF_ant2", 'r') as f2:
    chanF2_str = f2.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/chanF_ant3", 'r') as f3:
    chanF3_str = f3.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/gensym", 'r') as f4:
    gensym_str = f4.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/srs_delay", 'r') as f5:
    srsdelay_str = f5.read()


N_ifft = 1024
f_s = 30.72e6
Bandwidth = 18.36e6 # Occupied bandwidth is 18.36 MHz
N_ant = 4
osf = 8
srs_prb = 48
c = 299792458 # Speed of EM waves

# Edit these parameters based on the dataset collected
K = 486 # Total no. of times data is collected
idx_calibration = 100 # Index until which data is used for calibration
start_idx = 180 # Starting index for estimation
end_idx = 486 # Ending index for estimation
static_point = 286
visibility = np.concatenate([np.zeros(static_point), np.ones(end_idx - static_point)]) # visibility ground truth
D = 4.4 # UE distance from the BS
positions = np.concatenate([np.zeros([static_point, 2]), np.tile(np.array([[-2.15, 1.5]]), [end_idx - static_point, 1])], axis=0) # object position ground truth
AoA_ue = 21

chanF0 = np.zeros([K, N_ifft, 2])
chanF1 = np.zeros([K, N_ifft, 2])
chanF2 = np.zeros([K, N_ifft, 2])
chanF3 = np.zeros([K, N_ifft, 2])
gensym = np.zeros([K, N_ifft, 2])
srs_delay = np.zeros([K, N_ant])

chanstr2data(chanF0_str, N_ifft, K, chanF0)
chanstr2data(chanF1_str, N_ifft, K, chanF1)
chanstr2data(chanF2_str, N_ifft, K, chanF2)
chanstr2data(chanF3_str, N_ifft, K, chanF3)
chanstr2data(gensym_str, N_ifft, K, gensym)
delstr2data(srsdelay_str, K, srs_delay)

outliers = np.array(np.abs(srs_delay[:, 1] - srs_delay[:, 0]) > 32, dtype=int) + np.array(np.abs(srs_delay[:, 2] - srs_delay[:, 0]) > 32, dtype=int) + \
            np.array(np.abs(srs_delay[:, 3] - srs_delay[:, 0]) > 32, dtype=int) + np.array(np.abs(srs_delay[:, 0]) > 1000, dtype=int)

idx_outliers = [arg for arg in range(0, len(outliers)) if outliers[arg] != 0]

cal_indices = np.setdiff1d(np.arange(0, idx_calibration, 1), idx_outliers)

complex_chanF0 = chanF0[:, :, 0] + 1j*chanF0[:, :, 1]
complex_chanF1 = chanF1[:, :, 0] + 1j*chanF1[:, :, 1]
complex_chanF2 = chanF2[:, :, 0] + 1j*chanF2[:, :, 1]
complex_chanF3 = chanF3[:, :, 0] + 1j*chanF3[:, :, 1]
complex_gensym = gensym[:, :, 0] + 1j*gensym[:, :, 1]

# m = np.arange(0, 12*srs_prb, 1) # New and longer set of SRS symbols (ZC sequence)
# complex_gensym = np.exp(-1j*np.pi*m*(m + 2)/(len(m)))
# complex_gensym = np.concatenate([complex_gensym[0:270], np.zeros(N_ifft - 12*srs_prb), complex_gensym[270:12*srs_prb]])
# complex_gensym = np.reshape(complex_gensym, [1, N_ifft])
# complex_gensym = np.tile(complex_gensym, [K, 1])

complex_rxsym0 = complex_chanF0*complex_gensym
complex_rxsym0 = np.concatenate([complex_rxsym0[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym0[:, N_ifft//2:N_ifft]], axis=1)
complex_time0 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym0, axis=1)
complex_rxsym1 = complex_chanF1*complex_gensym
complex_rxsym1 = np.concatenate([complex_rxsym1[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym1[:, N_ifft//2:N_ifft]], axis=1)
complex_time1 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym1, axis=1)
complex_rxsym2 = complex_chanF2*complex_gensym
complex_rxsym2 = np.concatenate([complex_rxsym2[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym2[:, N_ifft//2:N_ifft]], axis=1)
complex_time2 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym2, axis=1)
complex_rxsym3 = complex_chanF3*complex_gensym
complex_rxsym3 = np.concatenate([complex_rxsym3[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym3[:, N_ifft//2:N_ifft]], axis=1)
complex_time3 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym3, axis=1)

complex_gensym = np.concatenate([complex_gensym[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_gensym[:, N_ifft//2:N_ifft]], axis=1)
time_gensym = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_gensym, axis=1)

offset_rxsym0 = complex_time0*np.conjugate(complex_time0)
offset_rxsym1 = complex_time1*np.conjugate(complex_time0)
offset_rxsym2 = complex_time2*np.conjugate(complex_time0)
offset_rxsym3 = complex_time3*np.conjugate(complex_time0)

cal_mean_rxsym0 = np.sum(offset_rxsym0[cal_indices, :], axis=1)/np.sum(offset_rxsym0[cal_indices, :] != 0, axis=1)
cal_mean_rxsym1 = np.sum(offset_rxsym1[cal_indices, :], axis=1)/np.sum(offset_rxsym1[cal_indices, :] != 0, axis=1)
cal_mean_rxsym2 = np.sum(offset_rxsym2[cal_indices, :], axis=1)/np.sum(offset_rxsym2[cal_indices, :] != 0, axis=1)
cal_mean_rxsym3 = np.sum(offset_rxsym3[cal_indices, :], axis=1)/np.sum(offset_rxsym3[cal_indices, :] != 0, axis=1)

cal_mean_arr = np.array([cal_mean_rxsym0, cal_mean_rxsym1, cal_mean_rxsym2, cal_mean_rxsym3])
cal_mean_arr = np.average(cal_mean_arr, axis=1)

cal_phases = np.angle(cal_mean_arr)
cal_phases = np.tile(np.reshape(cal_phases, [N_ant,1]), [1, osf*N_ifft])

# Antenna spacing from antenna 0 in wavelengths
d_ant = np.array([0, 0.5, 1, 1.5])

true_detection = 0
false_alarm = 0
total_detection = 0
gt_detection = np.sum(visibility)
err = np.array([])
threshold = 0.3

t_search = np.arange(-50, 50, 1)
theta_search = np.arange(-5*np.pi/12, 5*np.pi/12, 0.1)
fft_2d = np.zeros([end_idx - start_idx, len(theta_search), len(t_search)])

for iter0 in range(start_idx, end_idx):
    print("--------------------- Symbol no. {} ------------------------".format(iter0))
    est_mean_time0 = complex_time0[iter0, :]
    est_mean_time1 = complex_time1[iter0, :]
    est_mean_time2 = complex_time2[iter0, :]
    est_mean_time3 = complex_time3[iter0, :]

    mean_chan_subcarrier = np.concatenate([[est_mean_time0], [est_mean_time1], [est_mean_time2], [est_mean_time3]], axis=0)
    mean_chan_subcarrier = mean_chan_subcarrier*np.exp(-1j*cal_phases)

    for iter_t in range(0, len(t_search)):
        for iter_theta in range(0, len(theta_search)):
            combined_chan = np.sum(np.tile(np.reshape(np.exp(1j*2*np.pi*d_ant*np.sin(theta_search[iter_theta])), [N_ant, 1]), [1, osf*N_ifft])*mean_chan_subcarrier, axis=0)
            fft_2d[iter0-start_idx, iter_theta, iter_t] = np.abs(np.sum(np.roll(np.conj(time_gensym[iter0, :]), t_search[iter_t])*combined_chan))

    fft_2d[iter0-start_idx, :, :] = fft_2d[iter0-start_idx, :, :]/np.max(fft_2d[iter0-start_idx, :, :])

    if iter0 > static_point:
        max_value = 0
        for iter_max in range(start_idx, static_point):
            diff = fft_2d[iter0-start_idx, :, :] - fft_2d[iter_max-start_idx, :, :]
            if np.max(diff) > max_value:
                max_value = np.max(diff)
                max_idx = iter_max
        

        fft_diff = fft_2d[iter0-start_idx, :, :] - fft_2d[max_idx-start_idx, :, :]

        if np.max(fft_diff) > threshold:
            l, m = np.unravel_index(np.argmax(fft_diff), fft_diff.shape)

            l_los, m_los = np.unravel_index(np.argmax(fft_2d[iter0-start_idx, :, :]), fft_2d[iter0-start_idx, :, :].shape)

            total_detection = total_detection + 1
            print("Path with (AoA, ToF) = ({}, {}) is  estimated to be a dynamic object".format(theta_search[l]*180/np.pi, t_search[m]*1e9/(osf*f_s)))

            print(t_search[m_los]*1e9/(osf*f_s))
            print(theta_search[l_los]*180/np.pi)

            if np.abs(theta_search[l_los]*180/np.pi - AoA_ue) > 10:
                false_alarm = false_alarm + 1
            else:
                if visibility[iter0] == 0:
                    false_alarm = false_alarm + 1
                
                else:
                    x_est, y_est = est_pos(D, l_hat=t_search[m]*c/(osf*f_s) - t_search[m_los]*c/(osf*f_s), theta_hat=theta_search[l]-AoA_ue*np.pi/180)
                    pos_err = np.sqrt((x_est - positions[iter0, 0])**2 + (y_est - positions[iter0, 1])**2)

                    if pos_err < 3:
                        true_detection = true_detection + 1
                        err = np.append(err, pos_err)
                    else:
                        false_alarm = false_alarm + 1


print(total_detection)
print(true_detection)
print(false_alarm)
print(gt_detection)
print(np.sqrt(np.sum(err**2)/len(err)))
print(err)
np.save("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/2d_fft_err.npy", err)