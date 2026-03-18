import numpy as np
import scipy
import numpy.matlib as mtlb
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt


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


with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanT_ant0", 'r') as f0:
    chan0_str = f0.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanT_ant1", 'r') as f1:
    chan1_str = f1.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanT_ant2", 'r') as f2:
    chan2_str = f2.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanT_ant3", 'r') as f3:
    chan3_str = f3.read()

N_chan = 8192
K = 122
chan0 = np.zeros([K, N_chan, 2])
chan1 = np.zeros([K, N_chan, 2])
chan2 = np.zeros([K, N_chan, 2])
chan3 = np.zeros([K, N_chan, 2])

str2data(chan0_str, N_chan, K, chan0)
str2data(chan1_str, N_chan, K, chan1)
str2data(chan2_str, N_chan, K, chan2)
str2data(chan3_str, N_chan, K, chan3)

complex_chan0 = chan0[:, :, 0] + 1j*chan0[:, :, 1]
complex_chan1 = chan1[:, :, 0] + 1j*chan1[:, :, 1]
complex_chan2 = chan2[:, :, 0] + 1j*chan2[:, :, 1]
complex_chan3 = chan3[:, :, 0] + 1j*chan3[:, :, 1]

abs_chan0 = np.abs(complex_chan0)
abs_chan1 = np.abs(complex_chan1)
abs_chan2 = np.abs(complex_chan2)
abs_chan3 = np.abs(complex_chan3)
max0 = np.argmax(abs_chan0)
max1 = np.argmax(abs_chan1)
max2 = np.argmax(abs_chan2)
max3 = np.argmax(abs_chan3)



idx = 0
time = np.arange(N_chan - 1024, N_chan, 1)  #np.arange(0, 1024, 1) 
# plt.figure(1) 
# plt.semilogy(np.arange(0, 1024, 1), abs_chan0[idx, time])
# plt.figure(2)
# plt.semilogy(np.arange(0, 1024, 1), abs_chan1[idx, time])
# plt.figure(3)
# plt.semilogy(np.arange(0, 1024, 1), abs_chan2[idx, time])
# plt.figure(4)
# plt.semilogy(np.arange(0, 1024, 1), abs_chan3[idx, time])
# plt.legend(['0', '1', '2', '3'])


#----------------------------------------------------------------------------------------#
#--------------------------- Analyzing channel in freq domain ---------------------------#
#----------------------------------------------------------------------------------------#

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant0", 'r') as f0:
    data0_str = f0.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant1", 'r') as f1:
    data1_str = f1.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant2", 'r') as f2:
    data2_str = f2.read()

with open("/home/bkumar2/openairinterface5g_isac/Datasets/chanF_ant3", 'r') as f3:
    data3_str = f3.read()

N_data = 1024
data0 = np.zeros([K, N_data, 2])
data1 = np.zeros([K, N_data, 2])
data2 = np.zeros([K, N_data, 2])
data3 = np.zeros([K, N_data, 2])

str2data(data0_str, N_data, K, data0)
str2data(data1_str, N_data, K, data1)
str2data(data2_str, N_data, K, data2)
str2data(data3_str, N_data, K, data3)

complex_data0 = data0[:, :, 0] + 1j*data0[:, :, 1]
complex_data1 = data1[:, :, 0] + 1j*data1[:, :, 1]
complex_data2 = data2[:, :, 0] + 1j*data2[:, :, 1]
complex_data3 = data3[:, :, 0] + 1j*data3[:, :, 1]

# complex_data0 = np.fft.ifft(complex_data0)
# complex_data1 = np.fft.ifft(complex_data1)
# complex_data2 = np.fft.ifft(complex_data2)
# complex_data3 = np.fft.ifft(complex_data3)
# complex_data1 = np.roll(complex_data1, max0 - max1)
# complex_data2 = np.roll(complex_data2, max0 - max2)
# complex_data3 = np.roll(complex_data3, max0 - max3)

abs_chan0 = np.abs(complex_data0)
abs_chan1 = np.abs(complex_data1)
abs_chan2 = np.abs(complex_data2)
abs_chan3 = np.abs(complex_data3)
abs_chan0 = np.concatenate((abs_chan0[:, 718:1024], abs_chan0[:, 0:270]), axis=1)
abs_chan1 = np.concatenate((abs_chan1[:, 718:1024], abs_chan1[:, 0:270]), axis=1)
abs_chan2 = np.concatenate((abs_chan2[:, 718:1024], abs_chan2[:, 0:270]), axis=1)
abs_chan3 = np.concatenate((abs_chan3[:, 718:1024], abs_chan3[:, 0:270]), axis=1)

# freq = np.arange(0, np.shape(abs_chan0)[1], 1) #np.arange(N_chan - 1024, N_chan, 1)
# plt.figure(1)
# plt.semilogy(freq, abs_chan0[idx, :])
# plt.figure(2)
# plt.semilogy(freq, abs_chan1[idx, :])
# plt.figure(3)
# plt.semilogy(freq, abs_chan2[idx, :])
# plt.figure(4)
# plt.semilogy(freq, abs_chan3[idx, :])

phase_data0 = np.angle(complex_data0)
phase_data1 = np.angle(complex_data1)
phase_data2 = np.angle(complex_data2)
phase_data3 = np.angle(complex_data3)

phase_data0 = np.concatenate((phase_data0[:, 718:1024], phase_data0[:, 0:270]), axis=1)
phase_data1 = np.concatenate((phase_data1[:, 718:1024], phase_data1[:, 0:270]), axis=1)
phase_data2 = np.concatenate((phase_data2[:, 718:1024], phase_data2[:, 0:270]), axis=1)
phase_data3 = np.concatenate((phase_data3[:, 718:1024], phase_data3[:, 0:270]), axis=1)

phase_diff1 = np.angle(complex_data1*np.conjugate(complex_data0))
phase_diff2 = np.angle(complex_data2*np.conjugate(complex_data0))
phase_diff3 = np.angle(complex_data3*np.conjugate(complex_data0))

phase_diff1 = np.concatenate((phase_diff1[:, 718:1024], phase_diff1[:, 0:270]), axis=1)
phase_diff2 = np.concatenate((phase_diff2[:, 718:1024], phase_diff2[:, 0:270]), axis=1)
phase_diff3 = np.concatenate((phase_diff3[:, 718:1024], phase_diff3[:, 0:270]), axis=1)

idx1 = 0
idx2 = 2

plt.figure(5)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_diff1[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_diff1[idx2, 1:])
plt.legend(['1', '2'])

plt.figure(6)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_diff2[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_diff2[idx2, 1:])
plt.legend(['1', '2'])

plt.figure(7)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_diff3[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_diff3[idx2, 1:])
plt.legend(['1', '2'])

plt.figure(8)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data0[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data0[idx2, 1:])
plt.legend(['1', '2'])

plt.figure(9)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data1[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data1[idx2, 1:])
plt.legend(['1', '2'])

plt.figure(10)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data2[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data2[idx2, 1:])
plt.legend(['1', '2'])

plt.figure(11)
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data3[idx1, 1:])
plt.plot(np.arange(0, np.shape(phase_diff1)[1]-1, 1), phase_data3[idx2, 1:])
plt.legend(['1', '2'])

mean_ph_diff1 = np.average(phase_diff1, axis=1)
mean_ph_diff2 = np.average(phase_diff2, axis=1)
mean_ph_diff3 = np.average(phase_diff3, axis=1)

plt.figure(12)
plt.plot(np.arange(0, K, 1), mean_ph_diff1)
plt.plot(np.arange(0, K, 1), mean_ph_diff2)
plt.plot(np.arange(0, K, 1), mean_ph_diff3)
plt.legend(['1','2','3'])

plt.show()

for k in range(0, K):
    print("\n---------- Display the mean difference between the phase --------------")
    print(np.sum(phase_diff1[k, :])/np.sum(phase_diff1[k, :] != 0))
    print(np.sum(phase_diff2[k, :])/np.sum(phase_diff2[k, :] != 0))
    print(np.sum(phase_diff3[k, :])/np.sum(phase_diff3[k, :] != 0))


#----------------------------------------------------------------------------------------#
#----------------------------------- Phase calibration ----------------------------------#
#----------------------------------------------------------------------------------------#

