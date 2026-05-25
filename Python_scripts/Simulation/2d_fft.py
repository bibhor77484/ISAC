import numpy as np
import scipy
import numpy.matlib as mtlb
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
import time
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator

np.set_printoptions(suppress=True)
np.random.seed(100)


# Compute the CDF from data and x values (x values are the values at which we want to evaluate the CDF)
def cdf(err, x):
    CDF = np.zeros(len(x))
    for iter in range(0, len(x)):
        CDF[iter] = np.sum(err <= x[iter])/len(err)
    return CDF


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


# Function to estimate the full channel estimates from combed channel estimates using interpolation 
def interpolate_channel_estimates(H_combed, nPRB, N_ifft, K_TC):
    H_combed = np.concatenate([H_combed[N_ifft-6*nPRB:N_ifft], H_combed[0:6*nPRB]])
    K_srs = int(np.ceil(12*nPRB / K_TC))
    H = np.zeros([12*nPRB], dtype=np.complex128)
    n = 0

    if K_TC == 2:
        filt8_start = np.array([12288, 8192, 4096, 0, 0, 0, 0, 0])/16384
        filt8_end = np.array([4096, 8192, 12288, 16384])/16384
        filt8_middle2 = np.array([4096, 8192, 8192, 8192, 4096, 0, 0, 0])/16384
        filt8_middle4 = np.array([0, 0, 4096, 8192, 8192, 8192, 4096, 0])/16384

        for k in range(0, K_srs):
            if k == 0:
                H[n:n+len(filt8_start)] = H[n:n+len(filt8_start)] + H_combed[k*K_TC]*filt8_start
            elif k == K_srs-1:
                H[n:n+len(filt8_end)] = H[n:n+len(filt8_end)] + H_combed[k*K_TC]*filt8_end
            elif np.mod(k, 2) == 1:
                H[n:n+len(filt8_middle2)] = H[n:n+len(filt8_middle2)] + H_combed[k*K_TC]*filt8_middle2
            elif np.mod(k, 2) == 0:
                H[n:n+len(filt8_middle4)] = H[n:n+len(filt8_middle4)] + H_combed[k*K_TC]*filt8_middle4
                n = k*K_TC

    elif K_TC == 4:
        filt16_start = np.array([12288, 8192, 8192, 8192, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])/16384
        filt16_end = np.array([4096, 8192, 8192, 8192, 12288, 16384, 16384, 16384, 0, 0, 0, 0, 0, 0, 0, 0])/16384
        filt16_middle = np.array([4096, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096, 0, 0, 0, 0, 0, 0, 0])/16384

        for k in range(0, K_srs):
            if k == 0:
                H[n:n+len(filt16_start)] = H[n:n+len(filt16_start)] + H_combed[k*K_TC]*filt16_start
            elif k == K_srs-1:
                if n+len(filt16_end) > len(H):
                    H[n:len(H)] = H[n:len(H)] + H_combed[k*K_TC]*filt16_end[0:len(H)-n]
                else:
                    H[n:n+len(filt16_end)] = H[n:n+len(filt16_end)] + H_combed[k*K_TC]*filt16_end
            else:
                if n+len(filt16_middle) > len(H):
                    H[n:len(H)] = H[n:len(H)] + H_combed[k*K_TC]*filt16_middle[0:len(H)-n]
                else:
                    H[n:n+len(filt16_middle)] = H[n:n+len(filt16_middle)] + H_combed[k*K_TC]*filt16_middle
                n = k*K_TC

    elif K_TC == 8:
        filt16_start = np.array([12288, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096, 0, 0, 0, 0, 0, 0, 0])/16384
        filt16_end = np.array([4096, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 12288, 16384, 16384, 16384, 16384, 16384, 16384, 16384])/16384
        filt16_middle = np.array([4096, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096])/16384

        for k in range(0, K_srs):
            if k == 0:
                H[n:n+len(filt16_start)] = H[n:n+len(filt16_start)] + H_combed[k*K_TC]*filt16_start
            elif k == K_srs-1:
                if n+len(filt16_end) > len(H):
                    H[n:len(H)] = H[n:len(H)] + H_combed[k*K_TC]*filt16_end[0:len(H)-n]
                else:
                    H[n:n+len(filt16_end)] = H[n:n+len(filt16_end)] + H_combed[k*K_TC]*filt16_end
            else:
                if n+len(filt16_middle) > len(H):
                    H[n:len(H)] = H[n:len(H)] + H_combed[k*K_TC]*filt16_middle[0:len(H)-n]
                else:
                    H[n:n+len(filt16_middle)] = H[n:n+len(filt16_middle)] + H_combed[k*K_TC]*filt16_middle
                n = k*K_TC
            

    H = np.concatenate([H[6*nPRB:12*nPRB], np.zeros(N_ifft - 12*nPRB), H[0:6*nPRB]])
    
    return H


# Function to find LoS path using the estimated AoA and ToF
def find_los(t_est, theta_est, alpha_est, aoa_ue):
    alpha_max = np.max(np.abs(alpha_est))
    
    while(len(t_est) > 0): # Continue until there is a path left to process
        l_ue = np.argmin(t_est) # Find the path index with the minimum transmit delay
        if (np.abs(theta_est[l_ue] - aoa_ue) > 7.5) or (np.abs(alpha_est[l_ue]) < 0.5*alpha_max): # Ignore the path if it is not coming from the UE or if it is too weak. LoS from the UE should be one of the strongest paths if not the strongest.
            t_est = np.delete(t_est, l_ue)
            theta_est = np.delete(theta_est, l_ue)
            alpha_est = np.delete(alpha_est, l_ue)
        else:
            break

    return t_est, theta_est, alpha_est, l_ue


### Permutations of path indices for different values of existing paths and new paths to be allocated. 
# For example, permute3_2 gives all the permutations of 2 paths allocated to 3 existing paths. 
## List of all possible permutations
permute1_3 = np.array([[0,6,6],
                       [6,0,6],
                       [6,6,0]])

permute2_3 = np.array([[0,1,6],
                      [1,0,6],
                      [0,6,1],
                      [1,6,0],
                      [6,0,1],
                      [6,1,0]])

permute3_3 = np.array([[0,1,2],
                     [0,2,1],
                     [1,0,2], 
                     [1,2,0], 
                     [2,0,1], 
                     [2,1,0]])

permute4_3 = np.array([[0,1,2],
                     [0,2,1],
                     [1,0,2], 
                     [1,2,0], 
                     [2,0,1], 
                     [2,1,0], 
                     [3,1,2],
                     [3,2,1],
                     [3,0,2], 
                     [3,2,0], 
                     [3,0,1], 
                     [3,1,0], 
                     [0,3,2],
                     [0,3,1],
                     [1,3,2], 
                     [1,3,0], 
                     [2,3,1], 
                     [2,3,0], 
                     [0,1,3],
                     [0,2,3],
                     [1,0,3], 
                     [1,2,3], 
                     [2,0,3], 
                     [2,1,3]])

permute5_3 = np.array([[0,1,2],
                     [0,2,1],
                     [1,0,2],
                     [1,2,0],
                     [2,0,1],
                     [2,1,0],
                     [3,1,2],
                     [3,2,1],
                     [3,0,2],
                     [3,2,0],
                     [3,0,1],
                     [3,1,0],
                     [0,3,2],
                     [0,3,1],
                     [1,3,2],
                     [1,3,0],
                     [2,3,1],
                     [2,3,0],
                     [0,1,3],
                     [0,2,3],
                     [1,0,3],
                     [1,2,3],
                     [2,0,3],
                     [2,1,3],
                     [4,1,2],
                     [4,2,1],
                     [4,0,2],
                     [4,2,0],
                     [4,0,1],
                     [4,1,0],
                     [0,4,2],
                     [0,4,1],
                     [1,4,2],
                     [1,4,0],
                     [2,4,1],
                     [2,4,0],
                     [0,1,4],
                     [0,2,4],
                     [1,0,4],
                     [1,2,4],
                     [2,0,4],
                     [2,1,4],
                     [4,1,3],
                     [4,2,3],
                     [4,0,3],
                     [4,3,0],
                     [4,3,1],
                     [4,3,2],
                     [0,4,3],
                     [1,4,3],
                     [2,4,3],
                     [3,4,0],
                     [3,4,1],
                     [3,4,2],
                     [0,3,4],
                     [1,3,4],
                     [2,3,4],
                     [3,0,4],
                     [3,1,4],
                     [3,2,4]])

permute1_2 = np.array([[0,6],
                       [6,0]])

permute2_2 = np.array([[0,1],
                       [1,0]])

permute3_2 = np.array([[0,1],
                       [0,2],
                       [1,0],
                       [1,2],
                       [2,0],
                       [2,1]])

permute4_2 = np.array([[0,1],
                       [0,2],
                       [0,3],
                       [1,0],
                       [1,2],
                       [1,3],
                       [2,0],
                       [2,1],
                       [2,3],
                       [3,0],
                       [3,1],
                       [3,2]])

permute5_2 = np.array([[0,1],
                      [0,2],
                      [0,3],
                      [0,4],
                      [1,0],
                      [1,2],
                      [1,3],
                      [1,4],
                      [2,0],
                      [2,1],
                      [2,3],
                      [2,4],
                      [3,0],
                      [3,1],
                      [3,2],
                      [3,4],
                      [4,0],
                      [4,1],
                      [4,2],
                      [4,3]])


### Signal and system parameters
D = 11 # distance of the UE from the gNB (in m)
c = 299792458
f_c = 3.6192e9
bandwidth = 20e6
scs = 30e3
K_TC = 2 # This is the SRS transmission comb

if bandwidth == 20e6:
    f_s = 30.72e6
    N_ifft = 1024
    nPRB = 51
    N_cp = 72
    occ_bandwidth = 12*nPRB*scs
    print("Bandwidth is 20 MHz")

elif bandwidth == 40e6:
    f_s = 61.44e6
    N_ifft = 2048
    nPRB = 106
    N_cp = 144
    occ_bandwidth = 12*nPRB*scs
    print("Bandwidth is 40 MHz")

elif bandwidth == 80e6:
    f_s = 122.88e6
    N_ifft = 4096
    nPRB = 217
    N_cp = 288
    occ_bandwidth = 12*nPRB*scs
    print("Bandwidth is 80 MHz")

osf = 8
N_ant = 4
d_ant = 1/2*np.arange(0, N_ant, 1) # inter-element distance (in terms of wavelength)
P_t = 0.1 # Watt
K = 1.380649e-23
T = 298
noise_var = K*bandwidth*T


true_detection = 0
false_alarm = 0
total_detection = 0
gt_detection = 0
err = np.array([])
threshold = 0.2

trajectories = np.array(['Trajectory_1', 'Trajectory_2', 'Trajectory_3'])

n_loop = 1 # Monte carlo iterations

time1 = time.time() 
for iter_traj in range(0, len(trajectories)):

    print("Start of {}th trajectory".format(iter_traj))

    ### Channel parameters
    chan_params = np.load('/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Datasets/Open_scenario/' + trajectories[iter_traj] + '/ISAC_ray_tracing_chan.npy')
    positions = np.load('/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Datasets/Open_scenario/' + trajectories[iter_traj] + '/ISAC_locations.npy')
    obj_params = np.load('/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Datasets/Open_scenario/' + trajectories[iter_traj] + '/obj_params.npy')

    visibility = obj_params[:, 2] # This array indicates the ground truth of object visibility to the BS.
    gt_detection = gt_detection + np.sum(visibility)*n_loop

    K = 100
    tau = chan_params[:, 2, :] # Transmit delay in s 
    theta = chan_params[:, 0, :] # AoA in radians
    Doppler = chan_params[:, 1, :] # Doppler in Hz (converted from m/s)
    pathgain = chan_params[:, 3, :] # Pathgain in absolute value

    for k in range(0, K):
        order = np.argsort(pathgain[k, :])[::-1]
        tau[k, :] = tau[k, order]
        theta[k, :] = theta[k, order]
        Doppler[k, :] = Doppler[k, order]
        pathgain[k, :] = pathgain[k, order]

    # Repeat the same values K times for averaging (Monte carlo simulation)
    # tau = np.tile(np.reshape(tau, [1, len(tau)]), [K, 1])
    # theta = np.tile(np.reshape(theta, [1, len(theta)]), [K, 1])
    theta = theta + np.pi
    theta = np.pi/2 - theta
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    obj_params[:, 0] = np.pi/180*obj_params[:, 0] 
    obj_params[:, 0] = obj_params[:, 0] + np.pi
    obj_params[:, 0] = np.pi/2 - obj_params[:, 0]
    obj_params[:, 0] = (obj_params[:, 0] + np.pi) % (2 * np.pi) - np.pi
    obj_params[:, 0] = 180/np.pi*obj_params[:, 0]
    # Doppler = np.tile(np.reshape(Doppler, [1, len(Doppler)]), [K, 1])
    pathgain = P_t*pathgain/noise_var

    static_point = np.min([i for i in range(len(visibility)) if visibility[i] == 1])

    for iter_mc in range(0, n_loop):
        print("Start of {}th monte carlo iteration".format(iter_mc))

        # Generate the received signal

        # No. of paths
        L = np.shape(tau)[1]

        # Generated signal for K scenarios, N_ant antennas and N_ifft samples
        gen_signal = np.zeros([K, N_ant, N_ifft], dtype=np.complex128)

        # Transmitted OFDM symbols
        # Xn = np.random.randint(0, 4, [12*nPRB]) # QPSK symbols
        # Xn = np.exp(1j*(np.pi*Xn/2 + np.pi/4))

        Xn = np.zeros([12*nPRB], dtype=np.complex128)
        m = np.arange(0, int(np.ceil(12*nPRB / K_TC)), 1)
        Xn[np.arange(0, 12*nPRB, K_TC)] = np.exp(-1j*np.pi*m*(m + 2)/(len(m))) # SRS pilot symbols (ZC sequence)
        Xn[np.setdiff1d(np.arange(0, 12*nPRB, 1), np.arange(0, 12*nPRB, K_TC))] = np.exp(1j*(np.pi*np.random.randint(0, 4, [12*nPRB - len(m)])/2 + np.pi/4)) # Remaining subcarriers transmit QPSK symbols
        Xn = np.concatenate([Xn[6*nPRB:12*nPRB], np.zeros(N_ifft - 12*nPRB), Xn[0:6*nPRB]])
        Xn = np.reshape(Xn, [N_ifft, 1])

        X_pilot = np.zeros([12*nPRB], dtype=np.complex128)
        X_pilot[np.arange(0, 12*nPRB, K_TC)] = np.exp(-1j*np.pi*m*(m + 2)/(len(m)))
        X_pilot = np.concatenate([X_pilot[6*nPRB:12*nPRB], np.zeros(N_ifft - 12*nPRB), X_pilot[0:6*nPRB]])

        t = np.arange(N_cp, N_cp+N_ifft, 1)/f_s # Sampling timestamps, starts from N_cp because we discard the first N_cp samples
        n = np.concatenate([np.arange(0, N_ifft//2, 1), np.arange(-N_ifft//2, 0, 1)])
        n = np.reshape(n, [N_ifft, 1])
        n = np.tile(n, [1, N_ifft])

        # This loop just generates the OFDM samples in time domain
        for k in range(0, K):
            for l in range(0, L):
                if (theta[k, l] >= -np.pi/2) and (theta[k, l] < np.pi/2):
                    phase = np.exp(1j*2*np.pi*np.random.rand())
                    temp_sgnl = 1/(np.sqrt(N_ifft))*np.sqrt(pathgain[k, l]) * phase * \
                                np.exp(1j*2*np.pi*Doppler[k, l]*t) * \
                                np.sum(np.tile(Xn, [1, N_ifft])*np.exp(1j*2*np.pi*scs*n*np.tile(np.reshape(t-N_cp/f_s-tau[k, l], [1, N_ifft]), [N_ifft, 1])), axis=0)

                    gen_signal[k, :, :] = gen_signal[k, :, :] + np.tile(np.reshape(np.exp(1j*2*np.pi*d_ant*np.sin(theta[k, l])), [N_ant, 1]), [1, N_ifft]) * \
                                                                np.tile(np.reshape(temp_sgnl, [1, N_ifft]), [N_ant, 1])

        # Add the AWGN noise to the generated samples
        rcvd_signal = gen_signal + 1/np.sqrt(2)*(np.random.normal(size=[K, N_ant, N_ifft]) + 1j*np.random.normal(size=[K, N_ant, N_ifft]))

        # Convert the received samples back to frequency domain to estimate the OFDM symbols or channel
        rcvd_sig_freq = 1/np.sqrt(N_ifft)*np.fft.fft(rcvd_signal, axis=2)

        # Estimate the channel using X_pilot as pilot symbols
        H_combed = np.zeros([K, N_ant, N_ifft], dtype=np.complex128) 

        for i in range(0, N_ifft):
            if X_pilot[i] == 0:
                H_combed[:, :, i] = np.zeros([K, N_ant])
            else:
                H_combed[:, :, i] = rcvd_sig_freq[:, :, i]*np.conjugate(np.tile(X_pilot[i], [K, N_ant]))

        H = np.zeros([K, N_ant, N_ifft], dtype=np.complex128)
        for k in range(0, K):
            for n_ant in range(0, N_ant):
                H[k, n_ant, :] = interpolate_channel_estimates(H_combed[k, n_ant, :], nPRB, N_ifft, K_TC)


        L = 3 # No. of paths we are estimating

        track_params = np.zeros([10, 5, 2])
        track_idx = 0
        n_paths = np.zeros(np.shape(track_params)[1])

        m = np.arange(0, 12*nPRB, 1) # New and longer set of SRS symbols (ZC sequence)
        Xn = np.exp(-1j*np.pi*m*(m + 2)/(len(m)))
        Xn = np.concatenate([Xn[6*nPRB:12*nPRB], np.zeros(N_ifft - 12*nPRB), Xn[0:6*nPRB]])
        Xn = np.reshape(Xn, [N_ifft, 1])

        # First step is regenerating the received samples in time domain from the estimated channel and known pilot symbols (Xn) for all 4 different antennas
        complex_rxsym0 = H[:, 0, :]*np.tile(np.reshape(Xn[:, 0], [1, N_ifft]), [K, 1])
        complex_rxsym0 = np.concatenate([complex_rxsym0[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym0[:, N_ifft//2:N_ifft]], axis=1)
        complex_time0 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym0, axis=1)
        complex_rxsym1 = H[:, 1, :]*np.tile(np.reshape(Xn[:, 0], [1, N_ifft]), [K, 1])
        complex_rxsym1 = np.concatenate([complex_rxsym1[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym1[:, N_ifft//2:N_ifft]], axis=1)
        complex_time1 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym1, axis=1)
        complex_rxsym2 = H[:, 2, :]*np.tile(np.reshape(Xn[:, 0], [1, N_ifft]), [K, 1])
        complex_rxsym2 = np.concatenate([complex_rxsym2[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym2[:, N_ifft//2:N_ifft]], axis=1)
        complex_time2 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym2, axis=1)
        complex_rxsym3 = H[:, 3, :]*np.tile(np.reshape(Xn[:, 0], [1, N_ifft]), [K, 1])
        complex_rxsym3 = np.concatenate([complex_rxsym3[:, 0:N_ifft//2], np.zeros([K, (osf-1)*N_ifft]), complex_rxsym3[:, N_ifft//2:N_ifft]], axis=1)
        complex_time3 = np.sqrt(osf*N_ifft)*np.fft.ifft(complex_rxsym3, axis=1)

        # This is the IFFT samples of the OFDM symbols (This is later used in estimating the Doppler and ToA using Ambiguity function)
        Xn_os = np.concatenate([Xn[0:N_ifft//2, 0], np.zeros((osf-1)*N_ifft), Xn[N_ifft//2:N_ifft, 0]])
        time_gensym = np.sqrt(osf*N_ifft)*np.fft.ifft(Xn_os)

        t_search = np.arange(-50, 50, 1)
        theta_search = np.arange(-5*np.pi/12, 5*np.pi/12, 0.1)
        fft_2d = np.zeros([K, len(theta_search), len(t_search)])

        for iter0 in range(0, K):
            est_mean_time0 = complex_time0[iter0, :]
            est_mean_time1 = complex_time1[iter0, :]
            est_mean_time2 = complex_time2[iter0, :]
            est_mean_time3 = complex_time3[iter0, :]

            mean_chan_subcarrier = np.concatenate([[est_mean_time0], [est_mean_time1], [est_mean_time2], [est_mean_time3]], axis=0)

            print("-------------------- Symbol no. {} --------------------".format(iter0))

            for iter_t in range(0, len(t_search)):
                for iter_theta in range(0, len(theta_search)):
                    combined_chan = np.sum(np.tile(np.reshape(np.exp(-1j*2*np.pi*d_ant*np.sin(theta_search[iter_theta])), [N_ant, 1]), [1, osf*N_ifft])*mean_chan_subcarrier, axis=0)
                    fft_2d[iter0, iter_theta, iter_t] = np.abs(np.sum(np.roll(np.conj(time_gensym), t_search[iter_t])*combined_chan))

            fft_2d[iter0, :, :] = fft_2d[iter0, :, :]/np.max(fft_2d[iter0, :, :])

            if iter0 > static_point:
                max_value = 0
                for iter_max in range(0, static_point):
                    diff = fft_2d[iter0, :, :] - fft_2d[iter_max, :, :]
                    if np.max(diff) > max_value:
                        max_value = np.max(diff)
                        max_idx = iter_max

                fft_diff = fft_2d[iter0, :, :] - fft_2d[max_idx, :, :]

                if np.max(fft_diff) > threshold:
                    l, m = np.unravel_index(np.argmax(fft_diff), fft_diff.shape)

                    l_los, m_los = np.unravel_index(np.argmax(fft_2d[iter0, :, :]), fft_2d[iter0, :, :].shape)

                    print("Path with (AoA, ToF) = ({}, {}) is  estimated to be a dynamic object".format(theta_search[l]*180/np.pi, t_search[m]*1e9/(osf*f_s)))
                    
                    total_detection = total_detection + 1

                    if visibility[iter0] == 0:
                        false_alarm = false_alarm + 1
                    
                    else:
                        x_est, y_est = est_pos(D, l_hat=t_search[m]*c/(osf*f_s) - t_search[m_los]*c/(osf*f_s), theta_hat=theta_search[l])
                        pos_err = np.sqrt((x_est - positions[iter0, 0])**2 + (y_est - positions[iter0, 1])**2)

                        if pos_err < 5:
                            true_detection = true_detection + 1
                            err = np.append(err, pos_err)
                        else:
                            false_alarm = false_alarm + 1




time2 = time.time()

print(time2 - time1)

print(total_detection)
print(true_detection)
print(false_alarm)
print(gt_detection)
print(np.sqrt(np.sum(err**2)/len(err)))