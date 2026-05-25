import matplotlib.pyplot as plt
import numpy as np

# Data from the table
# SRS periodicity in ms (derived from 80ms, 40ms, 20ms, 10ms)
srs_ms = np.array([80, 40, 20, 10])

# Mean Throughput in Mbps
mean_throughput = np.array([1.9804, 1.9793, 1.963, 1.953])

# Standard Deviation in Mbps
sd_throughput = np.array([0.0775, 0.0884, 0.081, 0.093])

# Create the plot
plt.figure(figsize=(8, 5))

# Plot the mean line
plt.plot(srs_ms, mean_throughput, marker='o', linestyle='-', 
         color='b', label='Mean Throughput', linewidth=4)

# Add the standard deviation as a shaded area (error band)
# plt.fill_between(srs_ms, mean_throughput - sd_throughput, 
#                  mean_throughput + sd_throughput, 
#                  color='b', alpha=0.2, label='Standard Deviation')

# Formatting
# plt.title('Throughput vs. SRS Periodicity', fontsize=30)
plt.xlabel('SRS Periodicity (ms)', fontsize=40)
plt.ylabel('Throughput (Mbps)', fontsize=40)
plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=30)

# Set x-ticks to match the specific data points
plt.xticks(srs_ms, fontsize=40)
plt.yticks(fontsize=40)

# Adjust layout and show
plt.tight_layout()
plt.show()