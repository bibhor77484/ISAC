import numpy as np
import matplotlib.pyplot as plt


def cdf(err, x):
    CDF = np.zeros(len(x))
    for iter in range(0, len(x)):
        CDF[iter] = np.sum(err <= x[iter])/len(err)
    return CDF


x = np.arange(0, 5, 0.01)
Err1 = np.load("/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Error_indoor_detect.npy")
cdf1 = cdf(Err1, x)
Err2 = np.load("/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Error_open_detect.npy")
cdf2 = cdf(Err2, x)
# Err3 = np.load("/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Images/CDF_combsize/Error_ant4_BW20_obj1_interp_KTC4.npy")
# cdf3 = cdf(Err3, x)
# Err4 = np.load("/home/bkumar2/Documents/Research/ISAC-twin/Simulation_framework/Images/CDF_combsize/Error_ant4_BW20_obj1_interp_KTC8.npy")
# cdf4 = cdf(Err4, x)

plt.figure(0)
plt.plot(x, cdf1, c='b', linewidth=4)
plt.plot(x, cdf2, c='r', linewidth=4)
# plt.plot(x, cdf3, c='k', linewidth=4)
# plt.plot(x, cdf4, c='g', linewidth=4)
plt.xlabel("Error (m)", fontsize=40)
plt.ylabel("CDF", fontsize=40)
plt.grid(True)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.legend(['Indoor warehouse', 'Completely open environment'], fontsize=40)
plt.show()

# print(np.sqrt(1/len(Err1)*np.sum(Err1**2)))
# print(np.sqrt(1/len(Err2)*np.sum(Err2**2)))
# print(np.sqrt(1/len(Err3)*np.sum(Err3**2)))
# print(np.sqrt(1/len(Err4)*np.sum(Err4**2)))

# print(1/len(Err1)*np.sum(Err1))
# print(1/len(Err2)*np.sum(Err2))
# print(1/len(Err3)*np.sum(Err3))
# print(1/len(Err4)*np.sum(Err4))
# err_aoa_20 = np.load("Error_aoa_ant4_BW20_obj1.npy")
# x_aoa_20 = np.arange(0, np.max(err_aoa_20), 0.1)
# cdf1_20 = cdf(err_aoa_20, x_aoa_20)

# err_aoa_40 = np.load("Error_aoa_ant4_BW40_obj1.npy")
# x_aoa_40 = np.arange(0, np.max(err_aoa_40), 0.1)
# cdf1_40 = cdf(err_aoa_40, x_aoa_40)

# err_aoa_80 = np.load("Error_aoa_ant4_BW80_obj1.npy")
# x_aoa_80 = np.arange(0, np.max(err_aoa_80), 0.1)
# cdf1_80 = cdf(err_aoa_80, x_aoa_80)


# err_delay_20 = np.load("Error_delay_ant4_BW20_obj1.npy")
# x_delay_20 = np.arange(0, np.max(err_delay_20), 0.1)
# cdf2_20 = cdf(err_delay_20, x_delay_20)

# err_delay_40 = np.load("Error_delay_ant4_BW40_obj1.npy")
# x_delay_40 = np.arange(0, np.max(err_delay_40), 0.1)
# cdf2_40 = cdf(err_delay_40, x_delay_40)

# err_delay_80 = np.load("Error_delay_ant4_BW80_obj1.npy")
# x_delay_80 = np.arange(0, np.max(err_delay_80), 0.1)
# cdf2_80 = cdf(err_delay_80, x_delay_80)

# plt.figure(0)
# plt.plot(x_aoa_20, cdf1_20, c='b', linewidth=3)
# plt.plot(x_aoa_40, cdf1_40, c='k', linewidth=3)
# plt.plot(x_aoa_80, cdf1_80, c='r', linewidth=3)
# plt.xlabel("Error (degrees)", fontsize=30)
# plt.ylabel("CDF", fontsize=30)
# plt.legend(['20 MHz', '40 MHz', '80 MHz'], fontsize=30)
# plt.grid(True)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)

# plt.figure(1)
# plt.plot(x_delay_20, cdf2_20, c='b', linewidth=3)
# plt.plot(x_delay_40, cdf2_40, c='k', linewidth=3)
# plt.plot(x_delay_80, cdf2_80, c='r', linewidth=3)
# plt.xlabel("Error (ns)", fontsize=30)
# plt.ylabel("CDF", fontsize=30)
# plt.legend(['20 MHz', '40 MHz', '80 MHz'], fontsize=30)
# plt.grid(True)
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=30)
# plt.show()