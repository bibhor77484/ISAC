import numpy as np
import scipy
import numpy.matlib as mtlb
from scipy.optimize import root, minimize, least_squares
import matplotlib.pyplot as plt

def str2data(srs_delay_str, N_ant, K, srs_delay):
    delay = ''
    n = 0
    k = 0
    for i in range(0, len(srs_delay_str)):
        if(k == K):
            return
        
        if(srs_delay_str[i] == ','):
            srs_delay[n][k] = int(delay)
            delay = ''
            n = n+1
            continue
        
        if(srs_delay_str[i] == '\n'):
            srs_delay[n][k] = int(delay)
            delay = ''
            k = k+1
            n = 0
            continue

        delay = delay + srs_delay_str[i]


def func(var):
    x_ue, y_ue, z_ue = var
    # Example residuals
    r = 0
    for i in range(1, 4):
        r += (np.sqrt((x[i]-x_ue)**2 + (y[i]-y_ue)**2 + (z[i]-z_ue)**2) - c*delta_meas_delay[i-1] - np.sqrt((x[0]-x_ue)**2 + (y[0]-y_ue)**2 + (z[0]-z_ue)**2))**2
        
    return r


x = np.array([-0.65, -4.2, -0.85, -0.65])
y = np.array([-0.9, -1.33, -5, -1.99])
z = np.array([0, 0, 0, 0])

cal_SDO = np.array([-101.241564678544, -73.1454144074361, -48.1002246320682, -57.4432765298218])

# while (1):
N_ant = 4
K = 378
with open("/home/bkumar2/openairinterface5g_isac/Datasets/Same_location_6_sessions_6mins_gaps/srs_delay", 'r') as f:
    srs_delay_str = f.read()
f.close()

srs_delay = np.zeros([N_ant, K])
str2data(srs_delay_str, N_ant, K, srs_delay)

delta_srs_delay = srs_delay[1:4] - srs_delay[0]
plt.figure(1)
plt.plot(np.arange(0, K, 1), delta_srs_delay[0, :])
plt.plot(np.arange(0, K, 1), delta_srs_delay[1, :])
plt.plot(np.arange(0, K, 1), delta_srs_delay[2, :])
plt.legend(['Est. delay offset between Ant. 1 & 0', \
            'Est. delay offset between Ant. 2 & 0', \
            'Est. delay offset between Ant. 3 & 0'])
plt.xlabel('No. of measurements')
plt.ylabel("Time offset between antennas in ns")
plt.show()

meas_delay = np.array([-48.7142857142857, -34.9591836734694, 32.6034985422741, -7.53935860058309]) - cal_SDO

delta_meas_delay = meas_delay[1:4] - meas_delay[0]
delta_meas_delay = 1e-9*delta_meas_delay
# delta_meas_delay = np.mean(delta_meas_delay, 0)

c = 299792458

res = least_squares(func, [0,0,0])
ue_pos = res.x
# sol = minimize(func, x0=[0,0,0], method='Nelder-Mead', tol=1e-6) # Check this optimization method
print(meas_delay)

print("The position of the UE is [{}, {}, {}] \n".format(ue_pos[0], ue_pos[1], ue_pos[2]))
