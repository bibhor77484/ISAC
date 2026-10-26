import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

np.set_printoptions(suppress=True)
np.random.seed(10)


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


# Function to estimate the full channel estimates from combed channel estimates using interpolation (Support only for K_TC = 2 for now)
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
D = np.array([11, 11, 11]) # distance of the UE from the gNB (in m)
AoA_UE = np.array([0, 0, 0, 0, 0, 0, 0]) # AoA of the UE (in radians)
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
est_traj = np.zeros([3, 100, 2]) ### For saving the trajectory of the target

trajectories = np.array(['Trajectory_1', 'Trajectory_2', 'Trajectory_3'])

n_loop = 10 # Monte carlo iterations

for iter_traj in range(0, len(trajectories)):

    print("Start of {}th trajectory".format(iter_traj))

    # true_detection = 0
    # false_alarm = 0
    # total_detection = 0
    # gt_detection = 0
    # err = np.array([])

    ### Channel parameters
    chan_params = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/Datasets/Indoor_scenario/' + trajectories[iter_traj] + '/ISAC_ray_tracing_chan.npy')
    positions = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/Datasets/Indoor_scenario/' + trajectories[iter_traj] + '/ISAC_locations.npy')
    obj_params = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/Datasets/Indoor_scenario/' + trajectories[iter_traj] + '/obj_params.npy')

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

    time1 = time.time()
    ### Monte Carlo simulation to evaluate the performance of our sensing algorithm
    for iter_mc in range(0, n_loop):
        print("Start of {}th monte carlo iteration".format(iter_mc))

        dynamic_params = np.zeros([5, 2]) # This will store the parameters (range and angle) of a path if our logic deems it dynamic. Only one dynamic path is allowed.
        # static_params = np.array([]) # Define an empty array to store the parameters (range and angle) at the start of the experiment to save all possible static paths.
        
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
        temp_xn = np.tile(Xn, [1, N_ifft])
        phase = np.exp(1j*2*np.pi*np.random.rand(K, L))
        for k in range(0, K):
            for l in range(0, L):
                if ((theta[k, l] >= -np.pi/2) and (theta[k, l] < np.pi/2)) and (pathgain[k, l] > 1):
                    temp_sgnl = 1/(np.sqrt(N_ifft))*np.sqrt(pathgain[k, l]) * phase[k, l] * \
                                np.exp(1j*2*np.pi*Doppler[k, l]*t) * \
                                np.sum(temp_xn*np.exp(1j*2*np.pi*scs*n*np.tile(np.reshape(t-N_cp/f_s-tau[k, l], [1, N_ifft]), [N_ifft, 1])), axis=0)

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

        
        L = 3 # No. of paths we are estimating at each timestamp

        track_params = np.zeros([static_point, 5, 2])
        track_params_repo = np.zeros([5, 2])
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


        for iter0 in range(0, K):
            print("---------------------Start of {}th symbol------------------------".format(iter0))
            flag = 1

            est_mean_time0 = complex_time0[iter0, :]
            est_mean_time1 = complex_time1[iter0, :]
            est_mean_time2 = complex_time2[iter0, :]
            est_mean_time3 = complex_time3[iter0, :]

            mean_chan_subcarrier = np.concatenate([[est_mean_time0], [est_mean_time1], [est_mean_time2], [est_mean_time3]], axis=0)

            resolve = 5 # In mDTrack, the algorithm iteratively estimates the channel parameters. This variable sets the number of such iterations.
            parameters = np.zeros([4, L], dtype=np.complex128) # Save the estimated channel parameters in this array

            # Search space for the parameters
            f_search = np.arange(-200, 200, 100)
            t_search = np.arange(-50, 50, 1)
            theta_search = np.arange(-5*np.pi/12, 5*np.pi/12, 0.1)
            
            for iter in range(0, resolve):

                beam_strength = np.zeros([len(theta_search), L])
                caf_freq = np.zeros([len(f_search), L])
                caf_time = np.zeros([len(t_search), L])
                
                for l in range(0, L):

                    # After the first iteration, we refine the estimated parameters in the remaining iterations (Check mDTrack paper)
                    if iter > 0:
                        generated_sig = np.tile(np.reshape(time_gensym, [1,osf*N_ifft]), [N_ant,1])*np.tile(np.reshape(np.exp(1j*2*np.pi*d_ant*np.sin(np.real(parameters[0, l]))), [N_ant, 1]), [1, osf*N_ifft])
                        generated_sig = np.roll(generated_sig, int(np.real(parameters[2, l])), axis=1)
                        generated_sig = generated_sig*np.tile(np.reshape(np.exp(1j*2*np.pi*np.real(parameters[1, l])*n/(osf*f_s)), [1, osf*N_ifft]), [N_ant,1])
                        generated_sig = parameters[3, l]*generated_sig
                        mean_chan_subcarrier = generated_sig + residual_sig

                    if iter > 2:
                        theta_search = np.arange(-np.pi/24, np.pi/24, 0.01) + np.real(parameters[0, l])
                        
                    est_mean_rxsym = mean_chan_subcarrier*np.conjugate(np.tile(mean_chan_subcarrier[0:1, :], [N_ant,1]))

                    # mean_chan = np.average(est_mean_rxsym, axis=1)

                    # START - Code for beam sweeping here and finding the strongest direction.

                    for i in range(0, len(theta_search)):
                        beam_strength[i, l] = np.sum(np.abs(np.reshape(np.exp(-1j*2*np.pi*d_ant*np.sin(theta_search[i])), [1, N_ant]) @ est_mean_rxsym)**2)

                        # beam_strength[i, l] = np.abs(np.sum(mean_chan*np.exp(-1j*2*np.pi*d_ant*np.sin(theta_search[i]))))
                        
                    max_idx = np.argmax(beam_strength[:, l])
                    max_theta = theta_search[max_idx] 

                    parameters[0, l] = max_theta
                    # END - Code for beam sweeping here and finding the strongest direction.

                    # START - Code for estimating the transmit delay and Doppler using CAF (Cross Ambiguity function)

                    n = np.arange(0, osf*N_ifft, 1)

                    combined_chan = np.sum(np.tile(np.reshape(np.exp(-1j*2*np.pi*d_ant*np.sin(max_theta)), [N_ant, 1]), [1, osf*N_ifft])*mean_chan_subcarrier, axis=0)

                    max_caf = 0
                    for t in range(0, len(t_search)):
                        caf_time[t, l] = np.abs(np.sum(np.roll(np.conj(time_gensym), t_search[t])*combined_chan))
                        if caf_time[t,l] > max_caf:
                            t_best = t
                            max_caf = caf_time[t,l]
                        
                    max_caf = 0
                    shifted_combined_chan = np.roll(np.conj(time_gensym), t_search[t_best])*combined_chan
                    for f in range(0, len(f_search)):
                        caf_freq[f, l] = np.abs(np.sum(shifted_combined_chan*np.exp(-1j*2*np.pi*f_search[f]*n/(osf*f_s))))
                        if caf_freq[f,l] > max_caf:
                            f_best = f
                            max_caf = caf_freq[f,l]

                    # idx = np.argmax(caf[:, :, l])
                    # f_best, t_best = np.unravel_index(idx, caf[:, :, l].shape)

                    parameters[1, l] = f_search[f_best]
                    parameters[2, l] = t_search[t_best]

                    # END - Code for estimating the transmit delay and Doppler using CAF

                    # START - Code that is used to regenerate the signal using the estimated parameters

                    generated_sig = np.tile(np.reshape(time_gensym, [1,osf*N_ifft]), [N_ant,1])*np.tile(np.reshape(np.exp(1j*2*np.pi*d_ant*np.sin(max_theta)), [N_ant, 1]), [1, osf*N_ifft])

                    generated_sig = np.roll(generated_sig, t_search[t_best], axis=1)

                    generated_sig = generated_sig*np.tile(np.reshape(np.exp(1j*2*np.pi*f_search[f_best]*n/(osf*f_s)), [1, osf*N_ifft]), [N_ant,1])

                    alpha = np.average(mean_chan_subcarrier/generated_sig)

                    parameters[3, l] = alpha

                    generated_sig = alpha*generated_sig

                    residual_sig = mean_chan_subcarrier - generated_sig

                    mean_chan_subcarrier = residual_sig

                    # END - Code that is used to regenerate the signal using the estimated parameters


            # This part of the code will track the parameters - AoA and ToF
            if track_idx == 0: # First keep the first set of parameters in index 0 (l_idx indicates how many different sets of parameters are there in index 0)
                t_est = np.real(parameters[2, :])/(osf*f_s)*1e9
                theta_est = 180/np.pi*np.real(parameters[0, :])
                alpha_est = parameters[3, :]

                l = 0
                while(l < len(t_est)):
                    if np.abs(theta_est[l]) > 180/np.pi*(np.pi/2 - np.pi/24):
                        t_est = np.delete(t_est, l)
                        theta_est = np.delete(theta_est, l)
                        alpha_est = np.delete(alpha_est, l)
                        l = l-1
                    l = l+1

                t_est, theta_est, alpha_est, l_ue = find_los(t_est, theta_est, alpha_est, aoa_ue=AoA_UE[iter_traj])
                
                if len(t_est) == 0:
                    print("Skip this symbol. Cannot find the LoS path from UE") # Skip the processing of these estimates if we cannot find a good condidate for the LoS path from UE
                    continue

                t_est = t_est - t_est[l_ue]
                track_params[track_idx, 0:len(t_est), 0] = t_est    # track_params saves the valid paths for each symbol in a window after mapping them to track_params_repo.
                track_params[track_idx, 0:len(t_est), 1] = theta_est
                track_params_repo[0:len(t_est), 0] = t_est          # track_params_repo indicates the global repository of all the paths we are tracking. 
                track_params_repo[0:len(t_est), 1] = theta_est
                n_paths[0:len(t_est)] = n_paths[0:len(t_est)] + 1
                l_idx = len(t_est)                                  # l_idx indicates how many paths are there in track_params_repo.

                print("The estimated AoA of the paths are {}".format(np.real(parameters[0, :])*180/np.pi))
                print("The estimated delay of the paths are {}".format(np.real(parameters[2, :])*1e9/(osf*f_s)))

            else:
                t_est = np.real(parameters[2, :])/(osf*f_s)*1e9
                theta_est = 180/np.pi*np.real(parameters[0, :])
                alpha_est = parameters[3, :]

                l = 0
                while(l < len(t_est)):
                    if np.abs(theta_est[l]) > 180/np.pi*(np.pi/2 - np.pi/24):
                        t_est = np.delete(t_est, l)
                        theta_est = np.delete(theta_est, l)
                        alpha_est = np.delete(alpha_est, l)
                        l = l-1
                    l = l+1

                t_est, theta_est, alpha_est, l_ue = find_los(t_est, theta_est, alpha_est, aoa_ue=AoA_UE[iter_traj])
                
                if len(t_est) == 0:
                    print("Skip this symbol. Cannot find the LoS path from UE") # Skip the processing of these estimates if we cannot find a good condidate for the LoS path from UE
                    flag = 0

                if (flag == 1) and (iter0 < static_point): # 'flag' allows the processing of the estimates only if we can find a good condidate for the LoS path from UE
                    t_est = t_est - t_est[l_ue]

                    # This section of the code is used to find if the newly estimated parameters can be assigned to any of the existing paths (track_params_repo) or a new path. 
                    # We are using a simple threshold based method here for this assignment.
                    l = 0
                    while(l < len(t_est)):
                        dist = np.concatenate([np.reshape(np.abs(track_params_repo[0:l_idx, 0] - np.tile(t_est[l], l_idx)), [l_idx, 1]), np.reshape(np.abs(track_params_repo[0:l_idx, 1] - np.tile(theta_est[l], l_idx)), [l_idx, 1])], axis=1)
                        static_flag = dist < 15                     # static_flag contains a boolean array indicating if the newly estimated path is close enough to any of the existing paths in track_params_repo. 
                        
                        if np.max(np.sum(static_flag, axis=1)) < 2: # If the lth path is close to any of the existing paths in track_params_repo, then that corresponding row in static_flag will be both 1 and the sum will be 2. Otherwise, that path is new and needs to be added to track_params_repo.
                            if l_idx < 5:
                                track_params_repo[l_idx, 0] = t_est[l]
                                track_params_repo[l_idx, 1] = theta_est[l]
                                l_idx = l_idx + 1
                            else:
                                print("Path with (AoA, ToF) = ({}, {}) is dropped because the number of paths we are tracking is already 5.".format(theta_est[l], t_est[l]))
                                t_est = np.delete(t_est, l)
                                theta_est = np.delete(theta_est, l)
                                l = l - 1

                        l = l + 1
                                
                    
                    # Why does greedy not work here? - Because when doing greedy we may assign an outlier path to a track_params index which should have been assigned to the next 
                    # path. And we won't be able to assign the next path properly. 
                    # We are implementing brute force method by searching through all possible permutations because the number of paths is very small (3 to 5). 
                    # For larger number of paths, we can implement Hungarian algorithm or any other efficient algorithm for assignment problem.
                    
                    loss = np.zeros([7, len(t_est)])
                    loss[0:l_idx, 0:len(t_est)] = np.abs(np.tile(np.reshape(t_est, [1, len(t_est)]), [l_idx, 1]) - np.tile(track_params_repo[0:l_idx, 0:1], [1, len(t_est)])) + \
                        np.abs(np.tile(np.reshape(theta_est, [1, len(t_est)]), [l_idx, 1]) - np.tile(track_params_repo[0:l_idx, 1:2], [1, len(t_est)]))

                    if (l_idx == 1) and (len(t_est) == 3):  permute = permute1_3
                    if (l_idx == 2) and (len(t_est) == 3):  permute = permute2_3
                    if (l_idx == 3) and (len(t_est) == 3):  permute = permute3_3
                    if (l_idx == 4) and (len(t_est) == 3):  permute = permute4_3
                    if (l_idx == 5) and (len(t_est) == 3):  permute = permute5_3
                    if (l_idx == 1) and (len(t_est) == 2):  permute = permute1_2
                    if (l_idx == 2) and (len(t_est) == 2):  permute = permute2_2
                    if (l_idx == 3) and (len(t_est) == 2):  permute = permute3_2
                    if (l_idx == 4) and (len(t_est) == 2):  permute = permute4_2
                    if (l_idx == 5) and (len(t_est) == 2):  permute = permute5_2
                    if len(t_est) == 1:                     permute = np.reshape(np.arange(0, l_idx, 1), [l_idx, 1])

                    min_loss = 10000
                    min_idx = -1

                    for iter1 in range(0, np.shape(permute)[0]):
                        if len(t_est) == 3:
                            indices = ((permute[iter1, 0], permute[iter1, 1], permute[iter1, 2]), (0, 1, 2))
                        if len(t_est) == 2:
                            indices = ((permute[iter1, 0], permute[iter1, 1]), (0, 1))
                        if len(t_est) == 1:
                            indices = (permute[iter1, 0], 0)
                        
                        permuted_loss = np.sum(loss[indices])
                        
                        if permuted_loss < min_loss:
                            min_loss = permuted_loss
                            min_idx = iter1
                        
                    asgnmnt = permute[min_idx, :]

                    for l in range(0, len(t_est)):
                        if loss[asgnmnt[l], l] < 25: # This threshold is used to avoid assigning outliers to the existing paths.
                            if asgnmnt[l] != 6: # 6 means that this path is not allocated because track_params_repo is tracking fewer paths than we estimated this iteration.
                                track_params[track_idx, asgnmnt[l], 0] = t_est[l]
                                track_params[track_idx, asgnmnt[l], 1] = theta_est[l]
                                n_paths[asgnmnt[l]] = n_paths[asgnmnt[l]] + 1
                        else:
                            print("Path with estimated parameters (AoA, ToF) = ({}, {}) is not assigned to any existing path because it is an outlier.".format(theta_est[l], t_est[l]))

                    # print("The estimated AoA of the paths are {}".format(np.real(parameters[0, :])*180/np.pi))
                    # print("The estimated delay of the paths are {}".format(np.real(parameters[2, :])*1e9/(osf*f_s)))
                    # print(asgnmnt)
                    # print(n_paths)
                    # print(track_params_repo)

            # This scope detects the dynamic target(s) using the static paths saved earlier.
            if (iter0 >= static_point) and (flag == 1):
                t_est = t_est - t_est[l_ue]

                detect_flag = 0
                min_err = 100 ###

                for l in range(0, len(t_est)):
                    dist = np.concatenate([np.reshape(np.abs(static_params[:, 0] - np.tile(t_est[l], len(static_params))), [len(static_params), 1]), np.reshape(np.abs(static_params[:, 1] - np.tile(theta_est[l], len(static_params))), [len(static_params), 1])], axis=1)
                    static_flag = dist < 15
                    
                    if np.max(np.sum(static_flag, axis=1)) < 2:
                        print("Path with (AoA, ToF) = ({}, {}) is  estimated to be a dynamic object".format(theta_est[l], t_est[l]))
                        
                        total_detection = total_detection + 1

                        if visibility[iter0] == 0:
                            false_alarm = false_alarm + 1
                        
                        else:
                            x_est, y_est = est_pos(D[iter_traj], l_hat=t_est[l]*1e-9*c, theta_hat=theta_est[l]*np.pi/180-AoA_UE[iter_traj])
                            pos_err = np.sqrt((x_est - positions[iter0, 0])**2 + (y_est - positions[iter0, 1])**2)

                            if pos_err < 5:
                                detect_flag = 1
                                if pos_err < min_err:                                           ###
                                    min_err = pos_err                                           ###
                                    est_traj[iter_traj, iter0, :] = np.array([x_est, y_est])    ###
                            else:
                                false_alarm = false_alarm + 1
                
                if detect_flag == 1:
                    true_detection = true_detection + 1

            track_idx = track_idx + 1

            # This scope saves the static paths for the first static_point number of iterations where the environment is static.
            if (iter0 == static_point-1):
                est_mean_delay = np.sum(track_params[:, 0:l_idx, 0], axis=0)/n_paths[0:l_idx]
                est_mean_aoa = np.sum(track_params[:, 0:l_idx, 1], axis=0)/n_paths[0:l_idx]
                
                print("Estimated AoA of the static paths are {}".format(est_mean_aoa))
                print("Estimated delay of the static paths are {}".format(est_mean_delay))
        
                ### This part of the code is used to keep track of the static paths and use them to detect the dynamic path. This is a naive solution and can be improved.
                static_params = np.zeros([len(est_mean_delay), 2])
                static_params[:, 0] = est_mean_delay
                static_params[:, 1] = est_mean_aoa  
                            
                # Reinitialize the variables for next 10 iterations of tracking
                track_params = np.zeros([10, 5, 2])
                track_params_repo = np.zeros([5, 2])
                track_idx = 0
                n_paths = np.zeros(np.shape(track_params)[1])
            
            # This scope locates the dynamic target(s) using the static paths saved earlier. 
            # REPLACE THIS WITH KALMAN FILTER FOR JOURNAL WORK.
            elif ((track_idx == 10) and (iter0 >= static_point)): 
                est_mean_delay = np.sum(track_params[:, 0:l_idx, 0], axis=0)/n_paths[0:l_idx]
                est_mean_aoa = np.sum(track_params[:, 0:l_idx, 1], axis=0)/n_paths[0:l_idx]
                corr_paths = [i for i in range(0, len(n_paths)) if n_paths[i] >= np.max(n_paths)/2]

                # Reinitialize the variables for next 10 iterations of tracking
                track_params = np.zeros([10, 5, 2])
                track_params_repo = np.zeros([5, 2])
                track_idx = 0
                n_paths = np.zeros(np.shape(track_params)[1])

                final_est_aoa = est_mean_aoa[corr_paths]
                final_est_delay = est_mean_delay[corr_paths]

                dynamic_params_aoa = np.array([])
                dynamic_params_delay = np.array([])
                dynamic_pos_err = np.array([])

                for l in range(0, len(corr_paths)):
                    dist = np.concatenate([np.reshape(np.abs(static_params[:, 0] - np.tile(final_est_delay[l], len(static_params))), [len(static_params), 1]), np.reshape(np.abs(static_params[:, 1] - np.tile(final_est_aoa[l], len(static_params))), [len(static_params), 1])], axis=1)
                    static_flag = dist < 15
                    
                    if np.max(np.sum(static_flag, axis=1)) < 2:
                        # print("Path with (AoA, ToF) = ({}, {}) is  estimated to be a dynamic object".format(final_est_aoa[l], final_est_delay[l]))
                        dynamic_params_aoa = np.append(dynamic_params_aoa, final_est_aoa[l])
                        dynamic_params_delay = np.append(dynamic_params_delay, final_est_delay[l])

                        x_est, y_est = est_pos(D[iter_traj], l_hat=final_est_delay[l]*1e-9*c, theta_hat=final_est_aoa[l]*np.pi/180-AoA_UE[iter_traj])
                        pos_err = np.sqrt((x_est - np.average(positions[iter0+1-10:iter0+1, 0]))**2 + (y_est - np.average(positions[iter0+1-10:iter0+1, 1]))**2)

                        if pos_err < 5:
                            dynamic_pos_err = np.append(dynamic_pos_err, pos_err)


                if len(dynamic_pos_err) > 0:
                    err = np.append(err, np.min(dynamic_pos_err))

    # print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    # print("Total number of detections for {}: {}".format(trajectories[iter_traj], total_detection))
    # print("True detections for {}: {}".format(trajectories[iter_traj], true_detection))
    # print("False alarms for {}: {}".format(trajectories[iter_traj], false_alarm))
    # print("Ground truth detections for {}: {}".format(trajectories[iter_traj], gt_detection))
    # print("Root mean square error for {}: {}".format(trajectories[iter_traj], np.sqrt(np.sum(err**2)/len(err))))


print(total_detection)
print(true_detection)
print(false_alarm)
print(gt_detection)
print(np.sqrt(np.sum(err**2)/len(err)))

# np.save("est_traj_CellSense_Indoor.npy", est_traj)
np.save("Error_Indoor_CellSense.npy", err)