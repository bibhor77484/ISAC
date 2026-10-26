# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# # Data
# srs_ms = np.array([10, 20, 40, 80])
# mean_throughput = np.array([1.953, 1.963, 1.9793, 1.9804])
# sd_throughput = np.array([0.093, 0.081, 0.0884, 0.0775])

# fig, ax = plt.subplots(figsize=(10, 6))

# # 1. Plot the main data (with label for the main legend)
# line_plot = ax.plot(srs_ms, mean_throughput, marker='o', color='black', 
#                     label='Mean Throughput', linewidth=2, zorder=3)
# fill_plot = ax.fill_between(srs_ms, mean_throughput - sd_throughput, 
#                             mean_throughput + sd_throughput, 
#                             color='gray', alpha=0.3, label='Std. Deviation', zorder=2)

# # 2. Add sections without labels (keeps them out of the default legend)
# ax.axvspan(0, 30, color='red', alpha=0.1)
# ax.axvspan(30, 60, color='green', alpha=0.1)
# ax.axvspan(60, 100, color='blue', alpha=0.1)

# # 3. Create custom handles for the "Regions" legend
# red_patch = mpatches.Patch(color='red', alpha=0.1, label='Region A (Low)')
# green_patch = mpatches.Patch(color='green', alpha=0.1, label='Region B (Med)')
# blue_patch = mpatches.Patch(color='blue', alpha=0.1, label='Region C (High)')

# # 4. Create the first legend (Data) inside the plot
# leg1 = ax.legend(loc='upper left', frameon=True)

# # 5. Create the second legend (Regions) outside the plot
# # bbox_to_anchor=(1, 0.5) moves it to the right side
# leg2 = ax.legend(handles=[red_patch, green_patch, blue_patch], 
#                  title="Operational Regions",
#                  loc='center left', 
#                  bbox_to_anchor=(1, 0.5),
#                  frameon=True)

# # CRITICAL: Manually add the first legend back, otherwise it disappears
# ax.add_artist(leg1)

# # Formatting
# ax.set_xlabel('SRS Periodicity ($ms$)')
# ax.set_ylabel('Throughput ($Mbps$)')
# ax.grid(True, linestyle=':', alpha=0.6)
# ax.set_xlim(0, 90)

# # Use tight_layout to make room for the legend on the right
# plt.tight_layout()
# # plt.savefig('throughput_multi_legend.png', bbox_inches='tight')
# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# import matplotlib.pyplot as plt

# # Your data
# x_coords = [0, 0, -1.6, -1.6, 1.6, 2.9, 2.4, 1.8]
# y_coords = [4.4, 0, 1.8, 3.7, 2.15, 4.6, 2.7, 2.2]
# labels = ['UE', '5G gNB', '1', '2', '3', '4', '5', '6']
# markers = ['^', 'D', 'o', 'o', 'o', 'o', 'o', 'o']  

# plt.figure(figsize=(8, 5))

# # Loop through each point to plot it and add its text
# for i in range(len(x_coords)):
#     # 1. Plot the point
#     plt.plot(x_coords[i], y_coords[i], marker=markers[i], color='blue', markersize=20)
    
#     # 2. Add the label text
#     # xy=(x, y) is the point position
#     # xytext=(5, 5) moves the text 5 points to the right and 5 points up
#     plt.annotate(labels[i], 
#                  xy=(x_coords[i], y_coords[i]), 
#                  xytext=(7, 7), 
#                  textcoords='offset points',
#                  fontsize=40,
#                  fontweight='bold')

# plt.xlabel('X-Axis', fontsize=40)
# plt.ylabel('Y-Axis', fontsize=40)
# # plt.title('Points with Inline Labels', fontsize=40)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fontsize=40)
# plt.yticks(fontsize=40)

# # Expanding limits slightly so text doesn't get cut off at the edges
# plt.xlim(min(x_coords)-0.5, max(x_coords)+0.5)

# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# import matplotlib.pyplot as plt
# import numpy as np

# # Sample Data
# srs_ms = np.array([10, 20, 40, 80])
# throughput = np.array([1.953, 1.963, 1.9793, 1.9804])
# sensing = np.array([0.66, 0.74, 0.74, 0.74]) 

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # --- Plot 1: Throughput (Left Y-Axis) ---
# color1 = 'tab:blue'
# ax1.set_xlabel('SRS Periodicity ($ms$)', fontsize=40)
# ax1.set_ylabel('Throughput ($Mbps$)', color=color1, fontsize=40)
# line1 = ax1.plot(srs_ms, throughput, color=color1, marker='o', linewidth=5, label='Throughput')
# ax1.tick_params(axis='y', labelsize=40, labelcolor=color1)
# ax1.tick_params(axis='x', labelsize=40)
# ax1.grid(True, linestyle=':', alpha=0.6)

# # --- Plot 2: Latency (Right Y-Axis) ---
# # Create the twin axis
# ax2 = ax1.twinx() 

# color2 = 'tab:red'
# ax2.set_ylabel('Sensing accuracy ($meters$)', color=color2, fontsize=40)
# line2 = ax2.plot(srs_ms, sensing, color=color2, marker='s', linestyle='--', linewidth=5, label='Sensing accuracy')
# ax2.tick_params(axis='y', labelsize=40, labelcolor=color2)
# # ax2.tick_params(axis='x', labelsize=40)

# # --- Final Touches ---
# # Combine legends from both axes into one box
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left', fontsize=40)

# # plt.title('Dual-Axis Plot: Throughput and Sensing accuracy vs SRS Periodicity', fontsize=14)
# fig.tight_layout() # Ensures labels aren't cut off
# # plt.savefig('dual_axis_plot.png')

# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 1. Define the 10 physical experiment locations in the lab (X, Y in meters)
lab_x = np.array([-1.6, 1.6, 1.6, 2.9, 2.4, -1.8, -2.2, -2.7, 2.2, -2.15])
lab_y = np.array([1.8, 3.7, 2.15, 4.6, 2.7, 2.2, 2.65, 4, 2.65, 1.5])

# Fixed infrastructure locations
bs_coord = (0, 0)
ue_coord = (0, 4.4)

# 2. Generate dummy metrics for CellSense (Lower RMSE, Higher Probability)
# cellsense_rmse = np.array([1.52, 0.62, 1.85, 0.8, 0.62, 0.93, 2.58, 1.46, 0.77, 1.6])        # 10cm to 40cm error
cellsense_mean_err = np.array([1.5042064723853183, 0.637789290439579, 1.80288405085706, 0.8038808569669195, 0.5982698075294797, 0.902004794477962, 2.55, 1.4541511361101362, 0.7606828802617283, 1.4105405949446648])        # 10cm to 40cm error
cellsense_prob = np.array([133/145, 75/200, 192/200, 183/200, 196/200, 177/200, 87/200, 196/200, 114/200, 110/200])       # 85% to 100% detection

# 3. Generate dummy metrics for 2D-FFT (Higher RMSE, Lower Probability)
# fft_rmse = np.array([1.52, 1.98, 2.51, 3, 0.71, 2.84, 2.84, 1.77, 2.74, 2.13]) # 30cm to 90cm error
fft_mean_err = np.array([1.507711733520152, 1.9113905212003646, 2.48732810468034, 3, 0.7126833763162641, 2.8398746309439753, 2.74, 1.673132207843736, 2.728835246522078, 1.976159532839482]) # 30cm to 90cm error
fft_prob = np.array([129/145, 41/200, 96/200, 0/200, 9/200, 156/200, 22/200, 150/200, 67/200, 135/200]) # Lower probability

print(np.sum(cellsense_mean_err*cellsense_prob)/np.sum(cellsense_prob))
print(np.sum(fft_mean_err*fft_prob)/np.sum(fft_prob))

# Create a 2x2 subplot figure
fig, axs = plt.subplots(1, 2, figsize=(6, 10), sharey=True)

# Define circle sizes (s=300 makes them big and visible)
circle_size = 500 

# =========================================================================
# ROW 1: RMSE Comparison (Using '噴s' colormap: green is good, red is bad)
# =========================================================================
# vmin_rmse, vmax_rmse = 0.1, 3.0 # Standardize limits to compare fairly

# # CellSense RMSE
# sc1 = axs[0].scatter(lab_x, lab_y, c=cellsense_mean_err, s=circle_size, 
#                         cmap='RdYlGn_r', vmin=vmin_rmse, vmax=vmax_rmse, edgecolors='black')
# axs[0].set_title('CellSense', fontsize=50, fontweight='bold')

# # 2D-FFT RMSE
# axs[1].scatter(lab_x, lab_y, c=fft_mean_err, s=circle_size, 
#                   cmap='RdYlGn_r', vmin=vmin_rmse, vmax=vmax_rmse, edgecolors='black')
# axs[1].set_title('2D-FFT', fontsize=50, fontweight='bold')

# # Add a single colorbar for the RMSE row
# cbar_rmse = fig.colorbar(sc1, ax=axs, location='right', shrink=0.5, pad=0.03)
# cbar_rmse.ax.tick_params(labelsize=40)
# cbar_rmse.set_label('Mean localization error (m)', fontsize=50)

# =========================================================================
# ROW 2: Detection Probability Comparison (Using 'Blues' colormap)
# =========================================================================
vmin_prob, vmax_prob = 0, 1.0 

# CellSense Probability
sc2 = axs[0].scatter(lab_x, lab_y, c=cellsense_prob, s=circle_size, 
                        cmap='Blues', vmin=vmin_prob, vmax=vmax_prob, edgecolors='black')
axs[0].set_title('CellSense', fontsize=50, fontweight='bold')

# 2D-FFT Probability
axs[1].scatter(lab_x, lab_y, c=fft_prob, s=circle_size, 
                  cmap='Blues', vmin=vmin_prob, vmax=vmax_prob, edgecolors='black')
axs[1].set_title('2D-FFT', fontsize=50, fontweight='bold')

# Add a single colorbar for the Probability row
cbar_prob = fig.colorbar(sc2, ax=axs, location='right', shrink=0.5, pad=0.03)
cbar_prob.ax.tick_params(labelsize=40)
cbar_prob.set_label('Detection Probability', fontsize=50)

# =========================================================================
# Formatting the Lab Environment Layout
# =========================================================================
for ax in axs.flat:
    ax.set_xlim(-4, 4) # Assuming a 10m x 6m lab space
    ax.set_ylim(-0.4, 5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal') # Keeps the lab geometry from distorting
    ax.tick_params(labelsize=40)

    # Plot BS and UE with distinct shapes
    ax.scatter(bs_coord[0], bs_coord[1], color='black', marker='*', s=800, label='Base Station (BS)')
    ax.scatter(ue_coord[0], ue_coord[1], color='black', marker='D', s=500, label='User Equipment (UE)')

    # Add labels beside BS and UE (with a slight offset)
    ax.text(bs_coord[0] + 0.2, bs_coord[1], 'BS', fontsize=50, fontweight='bold', va='center')
    ax.text(ue_coord[0] + 0.2, ue_coord[1], 'UE', fontsize=50, fontweight='bold', va='center')

# Label the outer edges only
axs[0].set_xlabel('X-Coordinate (m)', fontsize=50)
axs[1].set_xlabel('X-Coordinate (m)', fontsize=50)
axs[0].set_ylabel('Y-Coordinate (m)', fontsize=50)
# axs[1, 0].set_ylabel('Y-Coordinate (m)', fontsize=50)

# plt.suptitle('Spatial Performance Mapping Across Lab Environment', fontsize=50, fontweight='bold', y=0.95)
# plt.savefig('spatial_performance_map.png', bbox_inches='tight', dpi=300)
plt.show()