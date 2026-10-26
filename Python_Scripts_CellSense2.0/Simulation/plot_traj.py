import numpy as np
import matplotlib.pyplot as plt


trajectories = np.array(['Trajectory_1', 'Trajectory_2'])

T_s = 0.02

x_visible = np.array([])
y_visible = np.array([])
x_invisible = np.array([])
y_invisible = np.array([])
est_traj_1 = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/est_traj_Outdoor_KF-CellSense.npy')
est_traj_2 = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/est_traj_Outdoor_KF_Doppler-CellSense_old.npy')
est_traj_3 = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/est_traj_Outdoor-CellSense.npy')

x_est_1 = np.array([])
y_est_1 = np.array([])
x_est_2 = np.array([])
y_est_2 = np.array([])
x_est_3 = np.array([])
y_est_3 = np.array([])

for i_traj in range(len(trajectories)):
    positions = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/Datasets/Outdoor_scenario/' + trajectories[i_traj] + '/ISAC_locations.npy')
    obj_params = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense2.0/Simulation/Datasets/Outdoor_scenario/' + trajectories[i_traj] + '/obj_params.npy')

    visibility = obj_params[:, 3]
    static_point = 30

    v0 = (positions[static_point+1, :] - positions[static_point, :])/T_s
    init_pos = positions[static_point, :] - v0*T_s*static_point
    
    for iter in range(0, len(positions)):

        if est_traj_1[i_traj, iter, 0] != 0:
            x_est_1 = np.append(x_est_1, est_traj_1[i_traj, iter, 0])
            y_est_1 = np.append(y_est_1, est_traj_1[i_traj, iter, 1])

        if est_traj_2[i_traj, iter, 0] != 0:
            x_est_2 = np.append(x_est_2, est_traj_2[i_traj, iter, 0])
            y_est_2 = np.append(y_est_2, est_traj_2[i_traj, iter, 1])

        if est_traj_3[i_traj, iter, 0] != 0:
            x_est_3 = np.append(x_est_3, est_traj_3[i_traj, iter, 0])
            y_est_3 = np.append(y_est_3, est_traj_3[i_traj, iter, 1])

        if iter < static_point:
            pos = init_pos + v0*T_s*iter
            x_invisible = np.append(x_invisible, pos[0])
            y_invisible = np.append(y_invisible, pos[1])

        elif iter >= static_point:
            pos = positions[iter, 0:2]

            if visibility[iter] == 1:
                x_visible = np.append(x_visible, pos[0])
                y_visible = np.append(y_visible, pos[1])

            else:
                x_invisible = np.append(x_invisible, pos[0])
                y_invisible = np.append(y_invisible, pos[1])

plt.scatter(x_invisible, y_invisible, color='r', marker='x', label='Object not visible', linewidths=5)
plt.scatter(x_visible, y_visible, color='b', marker='o', label='Visible object', linewidths=5)
plt.scatter(x_est_1, y_est_1, color='g', marker='*', label='Estimated positions (KF+CellSense)', linewidths=5)
plt.scatter(x_est_2, y_est_2, color='m', marker='s', label='Estimated positions (KF+Doppler+CellSense)', linewidths=5)
plt.scatter(x_est_3, y_est_3, color='k', marker='s', label='Estimated positions (CellSense)', linewidths=5)
plt.scatter(0, 0, color='k', marker='*', linewidths=50)
plt.annotate('gNB', xy=(0, 0), xytext=(40, 0), textcoords='offset points', fontsize=50)
plt.scatter(0, 20, color='k', marker='8', linewidths=50)
plt.annotate('UE', xy=(0, 20), xytext=(40, 0), textcoords='offset points', fontsize=50)
plt.xlabel('X-axis (m)', fontsize=60)
plt.ylabel('Y-axis (m)', fontsize=60)
plt.xticks(fontsize=60)
plt.yticks(fontsize=60)
plt.legend(loc='upper left', fontsize=50)
plt.grid(True)
plt.show()

print(np.sum(x_est_1 != 0))
print(np.sum(x_est_2 != 0))