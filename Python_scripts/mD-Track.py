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



with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant0", 'r') as f0:
    chanF0_str = f0.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant1", 'r') as f1:
    chanF1_str = f1.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant2", 'r') as f2:
    chanF2_str = f2.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant3", 'r') as f3:
    chanF3_str = f3.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/gensym", 'r') as f4:
    gensym_str = f4.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/srs_delay", 'r') as f5:
    srsdelay_str = f5.read()


N_chanF = 2048
f_s = 61.44e6
Bandwidth = 38.16e6 # Occupied bandwidth is 18.36 MHz
N_ant = 4

# Edit these parameters based on the dataset collected
K = 366 # Total no. of times data is collected
idx_calibration = 150 # Index until which data is used for calibration
idx_est1 = 214 # Starting index for estimation
idx_est2 = idx_est1 + 1 # Ending index for estimation
win_size = 1 # Window size for averaging 

f_search = np.arange(-10000, 10000, 1000)
t_search = np.arange(-50, 50, 1)
theta_search = np.arange(-np.pi/2, np.pi/2, 0.1)

chanF0 = np.zeros([K, N_chanF, 2])
chanF1 = np.zeros([K, N_chanF, 2])
chanF2 = np.zeros([K, N_chanF, 2])
chanF3 = np.zeros([K, N_chanF, 2])
gensym = np.zeros([K, N_chanF, 2])
srs_delay = np.zeros([K, N_ant])

chanstr2data(chanF0_str, N_chanF, K, chanF0)
chanstr2data(chanF1_str, N_chanF, K, chanF1)
chanstr2data(chanF2_str, N_chanF, K, chanF2)
chanstr2data(chanF3_str, N_chanF, K, chanF3)
chanstr2data(gensym_str, N_chanF, K, gensym)
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

complex_rxsym0 = complex_chanF0*complex_gensym
complex_time0 = np.fft.ifft(complex_rxsym0, axis=1)
complex_rxsym1 = complex_chanF1*complex_gensym
complex_time1 = np.fft.ifft(complex_rxsym1, axis=1)
complex_rxsym2 = complex_chanF2*complex_gensym
complex_time2 = np.fft.ifft(complex_rxsym2, axis=1)
complex_rxsym3 = complex_chanF3*complex_gensym
complex_time3 = np.fft.ifft(complex_rxsym3, axis=1)

temp_time_gensym = np.fft.ifft(complex_gensym, axis=1)
t = 1/f_s*np.concatenate([np.arange(0, N_chanF // 2, 1), np.arange(-N_chanF // 2, 0, 1)])
h = np.sinc(Bandwidth*t)
time_gensym_1 = circ_con(h, temp_time_gensym[0, :], N_chanF)
time_gensym = np.tile(time_gensym_1, [K, 1])

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
cal_phases = np.tile(np.reshape(cal_phases, [4,1]), [1, N_chanF])

time1 = time.time()

# Antenna spacing from antenna 0 in wavelengths
d = np.array([0, 0.5, 1, 1.5])

# est_mean_time0 = np.average(complex_time0[idx_est1:idx_est2, :], axis=0)
# est_mean_time1 = np.average(complex_time1[idx_est1:idx_est2, :], axis=0)
# est_mean_time2 = np.average(complex_time2[idx_est1:idx_est2, :], axis=0)
# est_mean_time3 = np.average(complex_time3[idx_est1:idx_est2, :], axis=0)

n_Dopp = np.concatenate([np.arange(0, N_chanF//2, 1), np.arange(0, N_chanF//2, 1)])

for iter0 in range(idx_est1, idx_est2):
    est_mean_time0 = complex_time0[iter0, :]
    est_mean_time1 = complex_time1[iter0, :]
    est_mean_time2 = complex_time2[iter0, :]
    est_mean_time3 = complex_time3[iter0, :]

    mean_chan_subcarrier = np.concatenate([[est_mean_time0], [est_mean_time1], [est_mean_time2], [est_mean_time3]], axis=0)
    mean_chan_subcarrier = mean_chan_subcarrier*np.exp(-1j*cal_phases)

    resolve = 5
    L = 3 # No. of paths
    parameters = np.zeros([4, L], dtype=np.complex128)

    print("-------------------- Symbol no. {} --------------------".format(iter0))
    
    for iter in range(0, resolve):
        # print("-------------------- Iteration no. {} --------------------".format(iter))

        beam_strength = np.zeros([len(np.arange(-np.pi/2, np.pi/2, 0.1)), L])
        caf = np.zeros([len(f_search), len(t_search), L])
        
        for l in range(0, L):

            if iter != 0:
                generated_sig = np.tile(time_gensym[0:1, :], [4,1])*np.tile(np.reshape(np.exp(1j*2*np.pi*d*np.sin(np.real(parameters[0, l]))), [4, 1]), [1, N_chanF])
                generated_sig = np.roll(generated_sig, int(np.real(parameters[2, l])))
                generated_sig = generated_sig*np.tile(np.reshape(np.exp(1j*2*np.pi*np.real(parameters[1, l])*n/f_s), [1, N_chanF]), [4,1])
                generated_sig = parameters[3, l]*generated_sig
                mean_chan_subcarrier = generated_sig + residual_sig

                theta_search = np.arange(-np.pi/20, np.pi/20, 0.01) + np.real(parameters[0, l])
                f_search = np.arange(-1000, 1000, 100) + np.real(parameters[1, l])

            est_mean_rxsym = mean_chan_subcarrier*np.conjugate(np.tile(mean_chan_subcarrier[0:1, :], [4,1]))

            mean_chan = np.average(est_mean_rxsym, axis=1)

            # START - Code for beam sweeping here and finding the strongest direction.

            for i in range(0, len(theta_search)):
                beam_strength[i, l] = np.abs(np.sum(mean_chan*np.exp(-1j*2*np.pi*d*np.sin(theta_search[i]))))
                
            plt.figure(2*L*iter + 2*l)
            plt.polar(theta_search, beam_strength[:, l])

            max_idx = np.argmax(beam_strength[:, l])
            max_theta = theta_search[max_idx] 

            parameters[0, l] = max_theta
            # END - Code for beam sweeping here and finding the strongest direction.

            time2 = time.time()

            # START - Code for estimating the transmit delay and Doppler using CAF

            n = np.concatenate([np.arange(0, N_chanF//2, 1), np.arange(0, N_chanF//2, 1)])
            
            for f in range(0, len(f_search)):
                for t in range(0, len(t_search)):
                    temp_caf = np.roll(np.conj(time_gensym[iter0, :]), t_search[t])*mean_chan_subcarrier[0, :]
                    caf[f, t, l] = caf[f, t, l] + 1/win_size*np.abs(np.sum(temp_caf*np.exp(-1j*2*np.pi*f_search[f]*n/f_s)))

            idx = np.argmax(caf[:, :, l])
            f_best, t_best = np.unravel_index(idx, caf[:, :, l].shape)

            parameters[1, l] = f_search[f_best]
            parameters[2, l] = t_search[t_best]

            # Moore's algorithm
            if iter == 0 and l == 0:
                fir_half = mean_chan_subcarrier[0, 0:(N_chanF//2)]
                sec_half = mean_chan_subcarrier[0, (N_chanF//2):N_chanF]
                epsilon = np.angle(fir_half*np.conjugate(time_gensym[0, 0:(N_chanF//2)]))

            # print(epsilon)
            # print(np.sum(sec_half - fir_half))

            # END - Code for estimating the transmit delay and Doppler using CAF

            # plt.figure(2*L*iter + 2*l+1)
            # plt.imshow(caf[:, :, l], cmap='hot', interpolation='nearest')
            # plt.xlabel("Delay sweep")
            # plt.ylabel("Doppler sweep")

            # START - Code that is used to regenerate the signal using the estimated parameters

            generated_sig = np.tile(time_gensym[0:1, :], [4,1])*np.tile(np.reshape(np.exp(1j*2*np.pi*d*np.sin(max_theta)), [4, 1]), [1, N_chanF])

            generated_sig = np.roll(generated_sig, t_search[t_best])

            generated_sig = generated_sig*np.tile(np.reshape(np.exp(1j*2*np.pi*f_search[f_best]*n/f_s), [1, N_chanF]), [4,1])

            alpha = np.average(mean_chan_subcarrier/generated_sig)

            parameters[3, l] = alpha

            generated_sig = alpha*generated_sig

            residual_sig = mean_chan_subcarrier - generated_sig

            mean_chan_subcarrier = residual_sig

            if iter == resolve-1:
                print("Path no. {}".format(l))
                print("Estimated Doppler is {} Hz".format(f_search[f_best]))
                print("Estimated delay is {} ns".format(t_search[t_best]/f_s*1e9))
                print("Estimated AoA is {}°".format(180/np.pi*max_theta))
                print("Estimated path gain is {}".format(alpha))

        # END - Code that is used to regenerate the signal using the estimated parameters

time3 = time.time()
# print("The timestamps for the program are = {}, {}, {}, {}".format(time0, time1, time2, time3))
# print(parameters)
plt.plot(np.arange(0, N_chanF//2, 1), epsilon)
plt.show()