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



with open("/home/bkumar2/openairinterface5g_isac/Datasets/Positioning/X310 Mar04/Different locations/chanF_ant0", 'r') as f0:
    chanF0_str = f0.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/Positioning/X310 Mar04/Different locations/chanF_ant1", 'r') as f1:
    chanF1_str = f1.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/Positioning/X310 Mar04/Different locations/chanF_ant2", 'r') as f2:
    chanF2_str = f2.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/Positioning/X310 Mar04/Different locations/chanF_ant3", 'r') as f3:
    chanF3_str = f3.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/Positioning/X310 Mar04/Different locations/srs_delay", 'r') as f4:
    srsdelay_str = f4.read()


N_chanF = 2048
osf = 8
f_s = 61.44e6
Bandwidth = 38.16e6 # Occupied bandwidth is 40 MHz
N_ant = 4

# Edit these parameters based on the dataset collected
K = 4508 # Total no. of times data is collected
K_cal = 150 # Index until which data is used for calibration
idx_est1 = 100 # Starting index for estimation
idx_est2 = idx_est1 + 1 # Ending index for estimation
win_size = 1 # Window size for averaging 

t_search = np.arange(-50, 50, 1)

chanF0 = np.zeros([K, N_chanF, 2])
chanF1 = np.zeros([K, N_chanF, 2])
chanF2 = np.zeros([K, N_chanF, 2])
chanF3 = np.zeros([K, N_chanF, 2])
srs_delay = np.zeros([K, N_ant])

chanstr2data(chanF0_str, N_chanF, K, chanF0)
chanstr2data(chanF1_str, N_chanF, K, chanF1)
chanstr2data(chanF2_str, N_chanF, K, chanF2)
chanstr2data(chanF3_str, N_chanF, K, chanF3)
delstr2data(srsdelay_str, K, srs_delay)

outliers = np.array(np.abs(srs_delay[:, 1] - srs_delay[:, 0]) > 200, dtype=int) + np.array(np.abs(srs_delay[:, 2] - srs_delay[:, 0]) > 200, dtype=int) + \
            np.array(np.abs(srs_delay[:, 3] - srs_delay[:, 0]) > 200, dtype=int) + np.array(np.abs(srs_delay[:, 0]) > 1000, dtype=int)

idx_outliers = [arg for arg in range(0, len(outliers)) if outliers[arg] != 0]

no_err_indices = np.setdiff1d(np.arange(0, K, 1), idx_outliers)

print(np.sum(no_err_indices < 330))

complex_chanF0 = chanF0[no_err_indices, :, 0] + 1j*chanF0[no_err_indices, :, 1]
complex_chanF1 = chanF1[no_err_indices, :, 0] + 1j*chanF1[no_err_indices, :, 1]
complex_chanF2 = chanF2[no_err_indices, :, 0] + 1j*chanF2[no_err_indices, :, 1]
complex_chanF3 = chanF3[no_err_indices, :, 0] + 1j*chanF3[no_err_indices, :, 1]
srs_delay = srs_delay[no_err_indices, :]

filtered_srs_delay = np.zeros(np.shape(srs_delay))

complex_chanF0 = np.concatenate([complex_chanF0[:, 0:N_chanF//2], np.zeros([np.shape(complex_chanF0)[0], (osf-1)*N_chanF]), complex_chanF0[:, N_chanF//2:N_chanF]], axis=1)
complex_chanF1 = np.concatenate([complex_chanF1[:, 0:N_chanF//2], np.zeros([np.shape(complex_chanF0)[0], (osf-1)*N_chanF]), complex_chanF1[:, N_chanF//2:N_chanF]], axis=1)
complex_chanF2 = np.concatenate([complex_chanF2[:, 0:N_chanF//2], np.zeros([np.shape(complex_chanF0)[0], (osf-1)*N_chanF]), complex_chanF2[:, N_chanF//2:N_chanF]], axis=1)
complex_chanF3 = np.concatenate([complex_chanF3[:, 0:N_chanF//2], np.zeros([np.shape(complex_chanF0)[0], (osf-1)*N_chanF]), complex_chanF3[:, N_chanF//2:N_chanF]], axis=1)

complex_time0 = np.fft.ifft(complex_chanF0, axis=1)
complex_time1 = np.fft.ifft(complex_chanF1, axis=1)
complex_time2 = np.fft.ifft(complex_chanF2, axis=1)
complex_time3 = np.fft.ifft(complex_chanF3, axis=1)

t = np.concatenate([np.arange(0, (osf*N_chanF) // 2, 1), np.arange(-(osf*N_chanF) // 2, 0, 1)])
h0 = np.sinc(Bandwidth*1/(osf*f_s)*t)

for iter0 in range(0, len(no_err_indices)):
    channel_time0 = complex_time0[iter0, :]
    channel_time1 = complex_time1[iter0, :]
    channel_time2 = complex_time2[iter0, :]
    channel_time3 = complex_time3[iter0, :]

    # plt.figure(0)
    # plt.plot(t, np.abs(channel_time0))
    # plt.figure(1)
    # plt.plot(t, np.abs(channel_time1))
    # plt.figure(2)
    # plt.plot(t, np.abs(channel_time2))
    # plt.figure(3)
    # plt.plot(t, np.abs(channel_time3))
    # plt.figure(4)
    # plt.plot(t, np.abs(h0))

    resolve = 5
    L = 3 # No. of paths
    Delay = np.zeros([N_ant, L], dtype=np.int32)
    Pathgain = np.zeros([N_ant, L], dtype=np.complex128)

    # print("-------------------- Symbol no. {} --------------------".format(iter0))
    
    for iter in range(0, resolve):
        # print("-------------------- Iteration no. {} --------------------".format(iter))
        
        for l in range(0, L):

            if iter != 0:
                generated_sig0 = Pathgain[0, l]*np.roll(h0, Delay[0, l])
                generated_sig1 = Pathgain[1, l]*np.roll(h0, Delay[1, l])
                generated_sig2 = Pathgain[2, l]*np.roll(h0, Delay[2, l])
                generated_sig3 = Pathgain[3, l]*np.roll(h0, Delay[3, l])

                channel_time0 = generated_sig0 + residual_sig0
                channel_time1 = generated_sig1 + residual_sig1
                channel_time2 = generated_sig2 + residual_sig2
                channel_time3 = generated_sig3 + residual_sig3

            time2 = time.time()

            # START - Code for estimating the transmit delay
            t_best = np.argmax(np.abs(channel_time0))
            Delay[0, l] = t[t_best]
            Pathgain[0, l] = channel_time0[t_best]/h0[0]

            t_best = np.argmax(np.abs(channel_time1))
            Delay[1, l] = t[t_best]
            Pathgain[1, l] = channel_time1[t_best]/h0[0]
            
            t_best = np.argmax(np.abs(channel_time2))
            Delay[2, l] = t[t_best]
            Pathgain[2, l] = channel_time2[t_best]/h0[0]
            
            t_best = np.argmax(np.abs(channel_time3))
            Delay[3, l] = t[t_best]
            Pathgain[3, l] = channel_time3[t_best]/h0[0]

            # END - Code for estimating the transmit delay

            # START - Code that is used to regenerate the signal using the estimated parameters
            generated_sig0 = np.roll(h0, Delay[0, l])
            generated_sig1 = np.roll(h0, Delay[1, l])
            generated_sig2 = np.roll(h0, Delay[2, l])
            generated_sig3 = np.roll(h0, Delay[3, l])

            generated_sig0 = Pathgain[0, l]*generated_sig0
            generated_sig1 = Pathgain[1, l]*generated_sig1
            generated_sig2 = Pathgain[2, l]*generated_sig2
            generated_sig3 = Pathgain[3, l]*generated_sig3

            residual_sig0 = channel_time0 - generated_sig0
            residual_sig1 = channel_time1 - generated_sig1
            residual_sig2 = channel_time2 - generated_sig2
            residual_sig3 = channel_time3 - generated_sig3

            channel_time0 = residual_sig0
            channel_time1 = residual_sig1
            channel_time2 = residual_sig2
            channel_time3 = residual_sig3

            # if (iter == resolve-1) and (l == L-1):
            #     # print("Path no. {}".format(l))
            #     # print("Estimated delay is {} ns".format(t_search[t_best]/f_s*1e9))
            #     # print("Estimated path gains are {}, {}, {}, {}".format(alpha0, alpha1, alpha2, alpha3))
            #     print(Delay[:, 0]/(osf*f_s)*1e9)

        # END - Code that is used to regenerate the signal using the estimated parameters
    filtered_srs_delay[iter0, :] = Delay[:, 0]/(osf*f_s)*1e9
    # filtered_srs_delay[iter0, :] = np.min(Delay, axis=1)/(osf*f_s)*1e9

# np.save(file='/home/bkumar2/openairinterface5g_isac/Datasets/Positioning/X310 Mar04/Different locations/Positioning/SingleP/COTSUE_diffloc_40M.npy', arr=filtered_srs_delay)

time3 = time.time()
# print("The timestamps for the program are = {}, {}, {}, {}".format(time0, time1, time2, time3))
# print(parameters)
# print(Delay/(osf*f_s)*1e9)
# print(Pathgain)


delta_filtered_srs_delay = filtered_srs_delay[:, 1:4] - np.tile(filtered_srs_delay[:, 0:1], [1, 3])
plt.figure(0)
plt.plot(np.arange(0, len(no_err_indices), 1), delta_filtered_srs_delay, linewidth=2)
plt.xlabel("SRS measurement snapshots", fontsize=24)
plt.ylabel("Delay in ns", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(['Diff. between Ant 1 and 0', 'Diff. between Ant 2 and 0', 'Diff. between Ant 3 and 0'], fontsize=30)
# plt.plot(np.array([280, 280]), np.array([-80, 80]), linewidth=3, linestyle='dashed')

delta_srs_delay = srs_delay[:, 1:4] - np.tile(srs_delay[:, 0:1], [1, 3])
plt.figure(1)
plt.plot(np.arange(0, len(no_err_indices), 1), delta_srs_delay, linewidth=2)
plt.xlabel("SRS measurement snapshots", fontsize=24)
plt.ylabel("Delay in ns", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(['Diff. between Ant 1 and 0', 'Diff. between Ant 2 and 0', 'Diff. between Ant 3 and 0'], fontsize=30)
plt.show()