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
        if (np.abs(theta_est[l_ue] - aoa_ue) > 15) or (np.abs(alpha_est[l_ue]) < 0.5*alpha_max): # Ignore the path if it is not coming from the UE or if it is too weak. LoS from the UE should be one of the strongest paths if not the strongest.
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



plt.ion()
fixed_font = 35

plt.rcParams.update({
    'font.size': fixed_font,          # Default font size
    'axes.titlesize': fixed_font,     # Plot titles
    'axes.labelsize': fixed_font,     # X and Y axis labels
    'xtick.labelsize': fixed_font,    # X-axis tick labels
    'ytick.labelsize': fixed_font,    # Y-axis tick labels
    'legend.fontsize': fixed_font     # Legend text
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 10))

# Plot 1
sc1 = ax1.scatter([], [], s=200)
ax1.set(xlim=(-5, 105), ylim=(-90, 90),
        title="Static paths", xlabel="Estimated ToF (ns)", ylabel="Estimated AoA (degrees)")
ax1.set_title("Static paths", pad=20)
ax1.grid(True)

# Plot 2
sc2 = ax2.scatter([], [], s=200)
sc4 = ax2.scatter([], [], color='red', s=200) # dynamic object
ax2.set(xlim=(-5, 105), ylim=(-90, 90),
        title="Current estimates (Red - target(s))", xlabel="Estimated ToF (ns)", ylabel="Estimated AoA (degrees)")
ax2.set_title("Current estimates (Red - target(s))", pad=20)
ax2.grid(True)

# Plot 3
sc3 = ax3.scatter([], [], color='red', s=200, zorder=10)
ax3.set(xlim=(-5, 5), ylim=(-0.5, 5),
        title="Estimated Positions (Red - target(s))", xlabel="X (m)", ylabel="Y (m)")
ax3.set_title("Estimated Positions (Red - target(s))", pad=20)
ax3.grid(True)

plt.tight_layout()
plt.show()

sc5 = ax3.scatter([0], [0], color='black', label='gNB', s=200, zorder=5)

ax3.annotate(
    'gNB',
    xy=(0, 0),
    xytext=(5, 5),
    textcoords='offset points',
    fontsize=fixed_font,
    zorder=4
)

sc5 = ax3.scatter([0], [4.4], color='black', label='UE', s=200, zorder=5)

ax3.annotate(
    'UE',
    xy=(0, 4.4),
    xytext=(5, 5),
    textcoords='offset points',
    fontsize=fixed_font,
    zorder=4
)

sc5 = ax3.scatter([1.8], [2.2], color='black', label='Target loc.', s=200, zorder=5)

ax3.annotate(
    'Target loc.',
    xy=(1.8, 2.2),
    xytext=(5, 5),
    textcoords='offset points',
    fontsize=fixed_font,
    zorder=4
)

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/6/chanF_ant0", 'r') as f0:
    chanF0_str = f0.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/6/chanF_ant1", 'r') as f1:
    chanF1_str = f1.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/6/chanF_ant2", 'r') as f2:
    chanF2_str = f2.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/6/chanF_ant3", 'r') as f3:
    chanF3_str = f3.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/6/gensym", 'r') as f4:
    gensym_str = f4.read()

with open("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/6/srs_delay", 'r') as f5:
    srsdelay_str = f5.read()


N_ifft = 1024
f_s = 30.72e6
Bandwidth = 18.36e6 # Occupied bandwidth is 18.36 MHz
N_ant = 4
osf = 8
srs_prb = 48
c = 299792458 # Speed of EM waves

# Edit these parameters based on the dataset collected
K = 460 # Total no. of times data is collected
idx_calibration = 100 # Index until which data is used for calibration
start_idx = 190 # Starting index for estimation
end_idx = 460 # Ending index for estimation
static_point = 260
visibility = np.concatenate([np.zeros(static_point), np.ones(end_idx - static_point)]) # visibility ground truth
D = 4.4 # UE distance from the BS
positions = np.concatenate([np.zeros([static_point, 2]), np.tile(np.array([[1.8, 2.2]]), [end_idx - static_point, 1])], axis=0) # object position ground truth
AoA_ue = 0

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

time1 = time.time()

# Antenna spacing from antenna 0 in wavelengths
d_ant = np.array([0, 0.5, 1, 1.5])

track_params = np.zeros([static_point-start_idx, 5, 2])
track_params_repo = np.zeros([5, 2])
track_idx = 0
n_paths = np.zeros(np.shape(track_params)[1])

true_detection = 0
false_alarm = 0
total_detection = 0
gt_detection = np.sum(visibility)
err = np.array([])

for iter0 in range(start_idx, end_idx):
    print("---------------------Start of {}th symbol------------------------".format(iter0))
    flag = 1

    est_mean_time0 = complex_time0[iter0, :]
    est_mean_time1 = complex_time1[iter0, :]
    est_mean_time2 = complex_time2[iter0, :]
    est_mean_time3 = complex_time3[iter0, :]

    mean_chan_subcarrier = np.concatenate([[est_mean_time0], [est_mean_time1], [est_mean_time2], [est_mean_time3]], axis=0)
    mean_chan_subcarrier = mean_chan_subcarrier*np.exp(-1j*cal_phases)

    resolve = 5
    L = 3 # No. of paths
    parameters = np.zeros([4, L], dtype=np.complex128)

    # Search space for the parameters
    f_search = np.arange(0, 1000, 1000)
    t_search = np.arange(-200, 200, 1)
    theta_search = np.arange(-5*np.pi/12, 5*np.pi/12, 0.1)

    print("-------------------- Symbol no. {} --------------------".format(iter0))
    
    for iter in range(0, resolve):
        # print("-------------------- Iteration no. {} --------------------".format(iter))

        beam_strength = np.zeros([len(theta_search), L])
        caf_freq = np.zeros([len(f_search), L])
        caf_time = np.zeros([len(t_search), L])
        
        for l in range(0, L):

            # After the first iteration, we refine the estimated parameters in the remaining iterations (Check mDTrack paper)
            if iter > 0:
                generated_sig = np.tile(np.reshape(time_gensym[iter0, :], [1,osf*N_ifft]), [N_ant,1])*np.tile(np.reshape(np.exp(-1j*2*np.pi*d_ant*np.sin(np.real(parameters[0, l]))), [N_ant, 1]), [1, osf*N_ifft])
                generated_sig = np.roll(generated_sig, int(np.real(parameters[2, l])), axis=1)
                generated_sig = generated_sig*np.tile(np.reshape(np.exp(1j*2*np.pi*np.real(parameters[1, l])*n/(osf*f_s)), [1, osf*N_ifft]), [N_ant,1])
                generated_sig = parameters[3, l]*generated_sig
                mean_chan_subcarrier = generated_sig + residual_sig

            if iter > 2:
                theta_search = np.arange(-np.pi/24, np.pi/24, 0.01) + np.real(parameters[0, l])
                # f_search = np.arange(-200, 200, 100) + np.real(parameters[1, l])

            est_mean_rxsym = mean_chan_subcarrier*np.conjugate(np.tile(mean_chan_subcarrier[0:1, :], [N_ant,1]))

            # mean_chan = np.average(est_mean_rxsym, axis=1)

            # START - Code for beam sweeping here and finding the strongest direction.

            for i in range(0, len(theta_search)):
                beam_strength[i, l] = np.sum(np.abs(np.reshape(np.exp(1j*2*np.pi*d_ant*np.sin(theta_search[i])), [1, N_ant]) @ est_mean_rxsym)**2)
                
                # np.abs(np.sum(mean_chan*np.exp(1j*2*np.pi*d_ant*np.sin(theta_search[i]))))
                
            # plt.figure(2*L*iter + 2*l)
            # plt.polar(theta_search, beam_strength[:, l])

            max_idx = np.argmax(beam_strength[:, l])
            max_theta = theta_search[max_idx] 

            parameters[0, l] = max_theta
            # END - Code for beam sweeping here and finding the strongest direction.

            time2 = time.time()

            # START - Code for estimating the transmit delay and Doppler using CAF (Cross Ambiguity function)

            n = np.arange(0, osf*N_ifft, 1)

            combined_chan = np.sum(np.tile(np.reshape(np.exp(1j*2*np.pi*d_ant*np.sin(max_theta)), [N_ant, 1]), [1, osf*N_ifft])*mean_chan_subcarrier, axis=0)
            # combined_chan = mean_chan_subcarrier[0, :]

            max_caf = 0
            for t in range(0, len(t_search)):
                caf_time[t, l] = np.abs(np.sum(np.roll(np.conj(time_gensym[iter0, :]), t_search[t])*combined_chan))
                if caf_time[t,l] > max_caf:
                    t_best = t
                    max_caf = caf_time[t,l]
                
            max_caf = 0
            shifted_combined_chan = np.roll(np.conj(time_gensym[iter0, :]), t_search[t_best])*combined_chan
            for f in range(0, len(f_search)):
                caf_freq[f, l] = np.abs(np.sum(shifted_combined_chan*np.exp(-1j*2*np.pi*f_search[f]*n/(osf*f_s))))
                if caf_freq[f,l] > max_caf:
                    f_best = f
                    max_caf = caf_freq[f,l]

            parameters[1, l] = f_search[f_best]
            parameters[2, l] = t_search[t_best]

            # END - Code for estimating the transmit delay and Doppler using CAF

            # plt.figure(2*L*iter + 2*l+1)
            # plt.imshow(caf[:, :, l], cmap='hot', interpolation='nearest')
            # plt.xlabel("Delay sweep")
            # plt.ylabel("Doppler sweep")

            # START - Code that is used to regenerate the signal using the estimated parameters

            generated_sig = np.tile(np.reshape(time_gensym[iter0, :], [1,osf*N_ifft]), [N_ant,1])*np.tile(np.reshape(np.exp(-1j*2*np.pi*d_ant*np.sin(max_theta)), [N_ant, 1]), [1, osf*N_ifft])

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

        print("The estimated AoA of the paths are {}".format(np.real(parameters[0, :])*180/np.pi))
        print("The estimated delay of the paths are {}".format(np.real(parameters[2, :])*1e9/(osf*f_s)))

        t_est, theta_est, alpha_est, l_ue = find_los(t_est, theta_est, alpha_est, aoa_ue=AoA_ue)
        
        if len(t_est) == 0:
            print("Skip this symbol. Cannot find the LoS path from UE") # Skip the processing of these estimates if we cannot find a good condidate for the LoS path from UE
            continue

        # if flag == 1:
        t_est = t_est - t_est[l_ue]
        track_params[track_idx, 0:len(t_est), 0] = t_est
        track_params[track_idx, 0:len(t_est), 1] = theta_est
        track_params_repo[0:len(t_est), 0] = t_est
        track_params_repo[0:len(t_est), 1] = theta_est
        n_paths[0:len(t_est)] = n_paths[0:len(t_est)] + 1
        l_idx = len(t_est)

    else:
        t_est = np.real(parameters[2, :])/(osf*f_s)*1e9
        theta_est = 180/np.pi*np.real(parameters[0, :])
        alpha_est = parameters[3, :]

        print("The estimated AoA of the paths are {}".format(np.real(parameters[0, :])*180/np.pi))
        print("The estimated delay of the paths are {}".format(np.real(parameters[2, :])*1e9/(osf*f_s)))

        t_est, theta_est, alpha_est, l_ue = find_los(t_est, theta_est, alpha_est, aoa_ue=AoA_ue)
        
        if len(t_est) == 0:
            print("Skip this symbol. Cannot find the LoS path from UE") # Skip the processing of these estimates if we cannot find a good condidate for the LoS path from UE
            flag = 0

        if (flag == 1) and (iter0 < static_point):
            t_est = t_est - t_est[l_ue]

            # This section of the code is used to find if the newly estimated parameters can be assigned to any of the existing paths (track_params) or a new path. 
            # We are using a simple threshold based method here for this assignment.
            l = 0
            while(l < len(t_est)):
                dist = np.concatenate([np.reshape(np.abs(track_params_repo[0:l_idx, 0] - np.tile(t_est[l], l_idx)), [l_idx, 1]), np.reshape(np.abs(track_params_repo[0:l_idx, 1] - np.tile(theta_est[l], l_idx)), [l_idx, 1])], axis=1)
                static_flag = dist < 15
                
                if np.max(np.sum(static_flag, axis=1)) < 2:
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
                    if asgnmnt[l] != 6: # This means that this path is not allocated because track_idx is tracking fewer paths than we estimated this iteration.
                        track_params[track_idx, asgnmnt[l], 0] = t_est[l]
                        track_params[track_idx, asgnmnt[l], 1] = theta_est[l]
                        n_paths[asgnmnt[l]] = n_paths[asgnmnt[l]] + 1
                else:
                    print("Path with estimated parameters (AoA, ToF) = ({}, {}) is not assigned to any existing path because it is an outlier.".format(theta_est[l], t_est[l]))

            # print("The estimated AoA of the paths are {}".format(np.real(parameters[0, :])*180/np.pi))
            # print("The estimated delay of the paths are {}".format(np.real(parameters[2, :])*1e9/(osf*f_s)))
            print(l_idx)
            print(loss)
            print(asgnmnt)
            print(n_paths)
            print(track_params_repo)

            sc1.set_offsets(np.column_stack((track_params_repo[0:l_idx, 0], track_params_repo[0:l_idx, 1])))
            fig.canvas.draw_idle()
            plt.pause(0.1)

    # This scope detects the dynamic paths using the static paths saved earlier.
    if (iter0 >= static_point) and (flag == 1):
        t_est = t_est - t_est[l_ue]

        l_dynamic = np.array([])

        detect_flag = 0
        for l in range(0, len(t_est)):
            dist = np.concatenate([np.reshape(np.abs(static_params[:, 0] - np.tile(t_est[l], len(static_params))), [len(static_params), 1]), np.reshape(np.abs(static_params[:, 1] - np.tile(theta_est[l], len(static_params))), [len(static_params), 1])], axis=1)
            static_flag = dist < 10
            
            if np.max(np.sum(static_flag, axis=1)) < 2:
                print("Path with (AoA, ToF) = ({}, {}) is  estimated to be a dynamic object".format(theta_est[l], t_est[l]))

                l_dynamic = np.append(l_dynamic, l)
                
                total_detection = total_detection + 1

                if visibility[iter0] == 0:
                    false_alarm = false_alarm + 1
                
                else:
                    x_est, y_est = est_pos(D, l_hat=t_est[l]*1e-9*c, theta_hat=(theta_est[l] - AoA_ue)*np.pi/180)
                    pos_err = np.sqrt((x_est - positions[iter0, 0])**2 + (y_est - positions[iter0, 1])**2)

                    if pos_err < 3:
                        detect_flag = 1
                    else:
                        false_alarm = false_alarm + 1

        sc2.set_offsets(np.column_stack((t_est, theta_est)))
        sc4.set_offsets(np.column_stack((t_est[l_dynamic.astype(int)], theta_est[l_dynamic.astype(int)])))
        fig.canvas.draw_idle()
        plt.pause(0.1)
        
        if detect_flag == 1:
            true_detection = true_detection + 1
        print("Total, true and false detections are {}, {}, {}".format(total_detection, true_detection, false_alarm))

    track_idx = track_idx + 1

    # This scope saves the static paths for the first static_point number of iterations where the environment is static.
    if (iter0 == static_point-1):
        est_mean_delay = np.sum(track_params[:, 0:l_idx, 0], axis=0)/n_paths[0:l_idx]
        est_mean_aoa = np.sum(track_params[:, 0:l_idx, 1], axis=0)/n_paths[0:l_idx]
        corr_paths = [i for i in range(0, len(n_paths)) if n_paths[i] >= np.max(n_paths)/4]

        ### This part of the code is used to keep track of the static paths and use them to detect the dynamic path. This is a naive solution and can be improved.
        static_params = np.zeros([len(corr_paths), 2])
        static_params[:, 0] = est_mean_delay[corr_paths]
        static_params[:, 1] = est_mean_aoa[corr_paths]

        sc1.set_offsets(np.column_stack((static_params[:, 0], static_params[:, 1])))
        fig.canvas.draw_idle()
        plt.pause(0.1)

        print("Estimated AoA of the static paths are {}".format(est_mean_aoa[corr_paths]))
        print("Estimated delay of the static paths are {}".format(est_mean_delay[corr_paths]))
        
        # Reinitialize the variables for next 10 iterations of tracking
        track_params = np.zeros([10, 5, 2])
        track_params_repo = np.zeros([5, 2])
        track_idx = 0
        n_paths = np.zeros(np.shape(track_params)[1])
    
    # This scope estimates the dynamic paths using the static paths saved earlier.
    elif ((track_idx == 2) and (iter0 >= static_point)): 
        est_mean_delay = np.sum(track_params[:, 0:l_idx, 0], axis=0)/n_paths[0:l_idx]
        est_mean_aoa = np.sum(track_params[:, 0:l_idx, 1], axis=0)/n_paths[0:l_idx]
        corr_paths = [i for i in range(0, len(n_paths)) if n_paths[i] > 0] #np.max(n_paths)/4]

        # Reinitialize the variables for next 10 iterations of tracking
        track_params = np.zeros([2, 5, 2])
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
            static_flag = dist < 10
            
            if np.max(np.sum(static_flag, axis=1)) < 2:
                dynamic_params_aoa = np.append(dynamic_params_aoa, final_est_aoa[l])
                dynamic_params_delay = np.append(dynamic_params_delay, final_est_delay[l])

                x_est, y_est = est_pos(D, l_hat=final_est_delay[l]*1e-9*c, theta_hat=(final_est_aoa[l] - AoA_ue)*np.pi/180)
                pos_err = np.sqrt((x_est - np.average(positions[iter0+1-10:iter0+1, 0]))**2 + (y_est - np.average(positions[iter0+1-10:iter0+1, 1]))**2)

                print("Estimated location of target ({}, {})".format(x_est, y_est))

                if pos_err < 3:
                    dynamic_pos_err = np.append(dynamic_pos_err, pos_err)
                
                sc3.set_offsets(np.column_stack((x_est, y_est)))
                fig.canvas.draw_idle()
                plt.pause(0.1)

        
        if len(dynamic_pos_err) > 0:
            err = np.append(err, np.min(dynamic_pos_err))

        # plt.figure((iter0+1) // 10)
        # plt.scatter(est_mean_aoa[corr_paths], est_mean_delay[corr_paths], c='b', marker='o', linewidth=10)
        # plt.xlabel("AoA (degrees)", fontsize=30)
        # plt.ylabel("Transmit delay (ns)", fontsize=30)
        # plt.grid(True)
        # plt.xticks(fontsize=30)
        # plt.yticks(fontsize=30)



        
time3 = time.time()
# print("The timestamps for the program are = {}, {}, {}, {}".format(time0, time1, time2, time3))
print("Execution time of the estimation and tracking code is {}".format(time3-time1))

# x = np.arange(0, np.max(err)+0.1, 0.1)
# cdf1 = cdf(err, x)

print(total_detection)
print(true_detection)
print(false_alarm)
print(gt_detection)
print(np.sqrt(np.sum(err**2)/len(err)))
print(static_params)
# np.save("/home/bkumar2/Documents/ISAC-main/ISAC/Datasets/CellSense_vs_2D-FFT/Fifth_attempt/12/CellSense_err.npy", err)

# plt.figure(0)
# plt.plot(x, cdf1, c='b', linewidth=3)
# plt.xlabel("Error (meters)", fontsize=30)
# plt.ylabel("CDF", fontsize=30)
# plt.grid(True)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.show()