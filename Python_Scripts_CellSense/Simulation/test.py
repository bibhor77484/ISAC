import numpy as np

file = [1, 2, 3, 4, 5, 6, 8, 9, 10]

for i in range(0, len(file)):
    err_cellsense = np.load('/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/' + str(file[i]) + '/CellSense_err.npy')
    err_2dfft = np.load('/home/bkumar2/Documents/Research/ISAC/Datasets/CellSense_vs_2D-FFT/Working_scenarios/' + str(file[i]) + '/2d_fft_err.npy')

    print("Mean error for CellSense and 2D-FFT are {} and {} respectively for scenario {}".format(np.mean(err_cellsense), np.mean(err_2dfft), file[i]))