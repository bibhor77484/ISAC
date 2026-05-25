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

N_chanF = 1024
K = 200
f_s = 30.72e6

f_search = np.arange(-500, 500, 50)
t_search = np.arange(-50, 50, 1)

loc1 = np.array([0, 100])
loc2 = np.array([140, 200])

chanF0 = np.zeros([K, N_chanF, 2])
chanF1 = np.zeros([K, N_chanF, 2])
chanF2 = np.zeros([K, N_chanF, 2])
chanF3 = np.zeros([K, N_chanF, 2])
gensym = np.zeros([K, N_chanF, 2])

str2data(chanF0_str, N_chanF, K, chanF0)
str2data(chanF1_str, N_chanF, K, chanF1)
str2data(chanF2_str, N_chanF, K, chanF2)
str2data(chanF3_str, N_chanF, K, chanF3)
str2data(gensym_str, N_chanF, K, gensym)

complex_chanF0 = chanF0[:, :, 0] + 1j*chanF0[:, :, 1]
complex_chanF1 = chanF1[:, :, 0] + 1j*chanF1[:, :, 1]
complex_chanF2 = chanF2[:, :, 0] + 1j*chanF2[:, :, 1]
complex_chanF3 = chanF3[:, :, 0] + 1j*chanF3[:, :, 1]
complex_gensym = gensym[:, :, 0] + 1j*gensym[:, :, 1]
complex_rxsym = complex_chanF0*complex_gensym

time_gensym = np.fft.ifft(complex_gensym, axis=1)
time_rxsym = np.fft.ifft(complex_rxsym, axis=1)

n = np.arange(0, N_chanF, 1)
caf = np.zeros([len(f_search), len(t_search)])
for i in range(0, 1):
    for f in range(0, len(f_search)):
        for t in range(0, len(t_search)):
            temp_caf = np.roll(np.conj(time_gensym[i, :]), t_search[t])*time_rxsym[i, :]
            caf[f, t] = np.abs(np.sum(temp_caf*np.exp(-1j*2*np.pi*f_search[f]*n/f_s)))

idx = np.argmax(caf)
f, t = np.unravel_index(idx, caf.shape)
print("Estimated Doppler is {} Hz".format(f_search[f]))
print("Estimated Doppler is {} ns".format(t_search[t]/f_s*1e9))
print(caf[:, t])

plt.imshow(caf, cmap='hot', interpolation='nearest')
plt.show()