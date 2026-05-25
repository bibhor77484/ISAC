import numpy as np
import matplotlib.pyplot as plt
import scipy


c = 3e8
f_c = 3.6192e9
bandwidth = 20e6
N_cp = 72
scs = 30e3

if bandwidth == 20e6:
    f_s = 30.72e6
    N_ifft = 1024
    nPRB = 51
    occ_bandwidth = 12*nPRB*scs
    
elif bandwidth == 40e6:
    f_s = 61.44e6
    N_ifft = 2048
    nPRB = 106
    occ_bandwidth = 12*nPRB*scs

osf = 8
N_ant = 4
d_ant = 1/2 # inter-element distance (in terms of wavelength)
P_t = 0.1 # Watt
K = 1.380649e-23
T = 298
noise_var = K*bandwidth*T
D = 10
tau = 1/c*np.array([0, 15, 25]) # Transmit delay in metres (converted into ns)
theta = np.pi/180*np.array([0, -40, 50]) # AoA in degree (converted into radians)
Doppler = f_c/c*np.array([0, 3, -2]) # Doppler in m/s (converted in Hz)
free_space = P_t*(c/f_c*1/(4*np.pi*10))**2
free_space = free_space/noise_var
pathgain = np.array([free_space, 0.2*free_space, 0.07*free_space])

L = len(tau)
gen_signal = np.zeros([N_ant, N_ifft], dtype=np.complex128)

Xn = np.random.randint(0, 4, [12*nPRB])
Xn = np.exp(1j*(np.pi*Xn/2 + np.pi/4))
Xn = np.concatenate([Xn[0:6*nPRB], np.zeros(N_ifft - 12*nPRB), Xn[6*nPRB:12*nPRB]])
Xn = np.reshape(Xn, [N_ifft, 1])

t = np.arange(N_cp, N_cp+N_ifft, 1)/f_s # Sampling timestamps
n = np.concatenate([np.arange(0, N_ifft//2, 1), np.arange(-N_ifft//2, 0, 1)])
n = np.reshape(n, [N_ifft, 1])
n = np.tile(n, [1, N_ifft])
for l in range(0, L):
    phase = np.exp(1j*2*np.pi*np.random.rand())
    temp_sgnl = np.sqrt(pathgain[l]) * phase * \
                np.exp(1j*2*np.pi*Doppler[l]*t) * \
                np.sum(np.tile(Xn, [1, N_ifft])*np.exp(1j*2*np.pi*scs*n*np.tile(np.reshape(t-N_cp/f_s-tau[l], [1, N_ifft]), [N_ifft, 1])), axis=0)

    gen_signal = gen_signal + np.tile(np.reshape(np.exp(1j*2*np.pi*d_ant*np.arange(0, N_ant, 1)*np.sin(theta[l])), [N_ant, 1]), [1, N_ifft])*np.tile(np.reshape(temp_sgnl, [1, N_ifft]), [N_ant, 1])

rcvd_signal = gen_signal + np.random.normal(size=[N_ant, N_ifft])

rcvd_sig_freq = np.fft.fft(rcvd_signal)
H = np.zeros(N_ifft, dtype=np.complex128)

for i in range(0, N_ifft):
    if Xn[i] == 0:
        H[i] = 0
    else:
        H[i] = rcvd_sig_freq[i]*np.conjugate(Xn[i, 0])

H = np.concatenate([H[0:N_ifft//2], np.zeros((osf-1)*N_ifft), H[N_ifft//2:N_ifft]])
cir = np.fft.ifft(H)
osf = 2
cir = np.concatenate([cir[-(osf*N_ifft)//2:], cir[0:(osf*N_ifft)//2]])

plt.semilogy(np.arange(-(osf*N_ifft)//2, (osf*N_ifft)//2, 1)/(8*f_s), np.abs(cir))
plt.xlabel("Time (us)", fontsize=30)
plt.ylabel("CIR", fontsize=30)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.show()