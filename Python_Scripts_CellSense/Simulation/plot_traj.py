import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

trajectories = np.array(['Trajectory_1', 'Trajectory_2'])

T_s = 0.08

x_visible = np.array([])
y_visible = np.array([])
x_invisible = np.array([])
y_invisible = np.array([])
est_traj = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense/Simulation/est_traj_Open.npy')

x_est = np.array([])
y_est = np.array([])

for i_traj in range(len(trajectories)):
    positions = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense/Simulation/Datasets/Open_scenario/' + trajectories[i_traj] + '/ISAC_locations.npy')
    obj_params = np.load('/home/bkumar2/Documents/Research/ISAC/Python_Scripts_CellSense/Simulation/Datasets/Open_scenario/' + trajectories[i_traj] + '/obj_params.npy')

    visibility = obj_params[:, 2]
    static_point = np.min([i for i in range(len(visibility)) if visibility[i] == 1])

    v0 = (positions[1, :] - positions[0, :])/T_s
    init_pos = positions[static_point, :] - v0*T_s*static_point
    
    for iter in range(0, len(positions)):

        if est_traj[i_traj, iter, 0] != 0:
            x_est = np.append(x_est, est_traj[i_traj, iter, 0])
            y_est = np.append(y_est, est_traj[i_traj, iter, 1])

        if iter < static_point:
            pos = init_pos + v0*T_s*iter
            x_invisible = np.append(x_invisible, pos[0])
            y_invisible = np.append(y_invisible, pos[1])

        elif iter >= static_point:
            pos = init_pos + v0*T_s*iter

            if visibility[iter] == 1:
                x_visible = np.append(x_visible, pos[0])
                y_visible = np.append(y_visible, pos[1])

            else:
                x_invisible = np.append(x_invisible, pos[0])
                y_invisible = np.append(y_invisible, pos[1])

plt.scatter(x_invisible, y_invisible, color='r', marker='x', label='Object not visible', linewidths=5)
plt.scatter(x_visible, y_visible, color='b', marker='o', label='Visible object', linewidths=5)
plt.scatter(x_est, y_est, color='g', marker='*', label='Estimated positions', linewidths=5)
plt.scatter(0, 0, color='k', marker='*', linewidths=1)
plt.scatter(0, 11, color='k', marker='*', linewidths=1)
plt.xlabel('X-axis (m)', fontsize=75)
plt.ylabel('Y-axis (m)', fontsize=75)
plt.xticks(fontsize=75)
plt.yticks(fontsize=75)
plt.legend(fontsize=75)
plt.grid(True)
plt.show()

print(np.sum(x_est != 0))