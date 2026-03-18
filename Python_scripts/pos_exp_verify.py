import numpy as np
import matplotlib.pyplot as plt


def est_pos(d, l_hat, theta_hat):
    x1 = (d + l_hat)*np.sin(theta_hat)
    y1 = (d + l_hat)*np.cos(theta_hat)
    m = (y1 - d)/x1

    x_mid = x1/2
    y_mid = (y1 + d)/2

    x_s_hat = (x_mid/m + y_mid)/(1/m + 1/np.tan(theta_hat))
    y_s_hat = x_s_hat/np.tan(theta_hat)

    return x_s_hat, y_s_hat


r_u = np.array([0, 10])

plt.figure(0)

plt.scatter(r_u[0], r_u[1], marker='.', linewidths=20)
plt.scatter(0, 0, marker='*', linewidths=20)
plt.grid(True)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylabel("Y-axis", fontsize=36)
plt.xlabel("X-axis", fontsize=36)

for iter in range(0, 10):
    r_s = np.array([20*np.random.random()-10, 20*np.random.random()])

    theta_hat = np.arctan(r_s[0]/r_s[1])
    theta_hat_err = theta_hat + 5/180*np.pi
    d = np.linalg.norm(r_u)
    l_hat = np.linalg.norm(r_s) + np.linalg.norm(r_s - r_u) - d
    l_hat_err = l_hat + 3

    x_s_hat, y_s_hat = est_pos(d, l_hat, theta_hat)
    x_s_hat_err, y_s_hat_err = est_pos(d, l_hat_err, theta_hat_err)

    plt.scatter(r_s[0], r_s[1], c='b', marker='o', linewidth=10)
    plt.scatter(x_s_hat, y_s_hat, c='r', marker='x', linewidth=10)
    plt.scatter(x_s_hat_err, y_s_hat_err, c='k', marker='.', linewidth=10)

    if iter == 0:
        plt.legend(['UE location', 'BS location', 'True scatterer locations', 'Analytically calculated scatterer locations'], fontsize=30)

plt.show()
