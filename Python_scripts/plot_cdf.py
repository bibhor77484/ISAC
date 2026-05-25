import numpy as np
import matplotlib.pyplot as plt


# Compute the CDF from data and x values (x values are the values at which we want to evaluate the CDF)
def cdf(err, x):
    CDF = np.zeros(len(x))
    for iter in range(0, len(x)):
        CDF[iter] = np.sum(err <= x[iter])/len(err)
    return CDF


CellSense_err = np.array([])
fft2d_err = np.array([])

for i_folder in range(1, 7):
    cellsense_err_temp = np.load("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/{}/CellSense_err.npy".format(i_folder))
    if len(cellsense_err_temp) > 0:
        CellSense_err = np.append(CellSense_err, cellsense_err_temp)

    fft2d_err_temp = np.load("/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/{}/2d_fft_err.npy".format(i_folder))
    if len(fft2d_err_temp) > 0:
        fft2d_err = np.append(fft2d_err, fft2d_err_temp)
    

x = np.arange(0, np.max(np.array([np.max(fft2d_err), np.max(CellSense_err)]))+0.2, 0.01)
CellSense_cdf = cdf(CellSense_err, x)
fft2d_cdf = cdf(fft2d_err, x)

plt.plot(x, CellSense_cdf, c='b', linewidth=5, label='CellSense')
plt.plot(x, fft2d_cdf, c='r', linewidth=5, label='2D-FFT')
plt.ylabel('CDF', fontsize=40)
plt.xlabel('Error (m)', fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.legend(fontsize=40)
plt.show()