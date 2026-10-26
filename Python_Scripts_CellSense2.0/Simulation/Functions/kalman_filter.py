import numpy as np

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


## Kalman Filter for tracking the location of the object of interest
def kalman_filter_wo_doppler(x, P, dynamic_params, D, AoA_UE, timestamps, l):
    R = np.array([[10, 0], [0, 10]])
    Q = np.array([[0.1, 0], [0, 0.1]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    c = 299792458
    
    if (timestamps[0, l] == -1) or (timestamps[1, l] == -1) or (timestamps[1, l] == timestamps[0, l]):
        return x, P

    y = np.zeros([2, 1])
    y[0, 0], y[1, 0] = est_pos(D, l_hat=dynamic_params[1, l, 0]*1e-9*c, theta_hat=dynamic_params[1, l, 1]*np.pi/180-AoA_UE)

    dt = (timestamps[1, l] - timestamps[0, l])

    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    G = np.array([[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt]])

    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), np.transpose(F)) + np.dot(np.dot(G, Q), np.transpose(G))

    S = np.dot(np.dot(H, P_pred), np.transpose(H)) + R
    K = np.dot(np.dot(P_pred, np.transpose(H)), np.linalg.inv(S))

    x_new = x_pred + np.dot(K, y - np.dot(H, x_pred))
    P_new = np.dot(np.eye(4) - np.dot(K, H), P_pred)

    return x_new, P_new


## Kalman Filter for tracking the location of the object of interest
def kalman_filter_w_doppler(x, P, dynamic_params, D, AoA_UE, timestamps, l, f_c, c):
    R = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0.25]])
    Q = np.array([[0.1, 0], [0, 0.1]])
    r_ue = np.array([[0], [D]])
    r_bs = np.array([[0], [0]])
    
    if (timestamps[0, l] == -1) or (timestamps[1, l] == -1) or (timestamps[1, l] == timestamps[0, l]):
        return x, P

    y = np.zeros([3, 1])
    y[0, 0], y[1, 0] = est_pos(D, l_hat=dynamic_params[1, l, 0]*1e-9*c, theta_hat=dynamic_params[1, l, 1]*np.pi/180-AoA_UE)
    y[2, 0] = c/f_c*dynamic_params[1, l, 2] # Doppler shift

    dt = (timestamps[1, l] - timestamps[0, l])

    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    G = np.array([[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt]])
    
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), np.transpose(F)) + np.dot(np.dot(G, Q), np.transpose(G))

    h1 = (np.linalg.norm(r_ue-x_pred[0:2, :])*(-x_pred[2,0]) - np.dot(np.transpose(x_pred[2:4, :]), r_ue-x_pred[0:2, :])*(x_pred[0,0]-r_ue[0,0])/np.linalg.norm(r_ue-x_pred[0:2, :]))/np.linalg.norm(r_ue-x_pred[0:2,:])**2 + (np.linalg.norm(r_bs-x_pred[0:2, :])*(-x_pred[2,0]) - np.dot(np.transpose(x_pred[2:4, :]), r_bs-x_pred[0:2, :])*(x_pred[0,0]-r_bs[0,0])/np.linalg.norm(r_bs-x_pred[0:2, :]))/np.linalg.norm(r_bs-x_pred[0:2,:])**2
    h2 = (np.linalg.norm(r_ue-x_pred[0:2, :])*(-x_pred[3,0]) - np.dot(np.transpose(x_pred[2:4, :]), r_ue-x_pred[0:2, :])*(x_pred[1,0]-r_ue[1,0])/np.linalg.norm(r_ue-x_pred[0:2, :]))/np.linalg.norm(r_ue-x_pred[0:2,:])**2 + (np.linalg.norm(r_bs-x_pred[0:2, :])*(-x_pred[3,0]) - np.dot(np.transpose(x_pred[2:4, :]), r_bs-x_pred[0:2, :])*(x_pred[1,0]-r_bs[1,0])/np.linalg.norm(r_bs-x_pred[0:2, :]))/np.linalg.norm(r_bs-x_pred[0:2,:])**2
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [h1[0,0], \
                                               h2[0,0], \
                                               (r_ue[0,0]-x_pred[0,0])/np.linalg.norm(r_ue-x_pred[0:2, :]) + (r_bs[0, 0]-x_pred[0,0])/np.linalg.norm(r_bs-x_pred[0:2, :]), \
                                               (r_ue[1,0]-x_pred[1,0])/np.linalg.norm(r_ue-x_pred[0:2, :]) + (r_bs[1, 0]-x_pred[1,0])/np.linalg.norm(r_bs-x_pred[0:2, :])]])

    S = np.dot(np.dot(H, P_pred), np.transpose(H)) + R
    K = np.dot(np.dot(P_pred, np.transpose(H)), np.linalg.inv(S))

    y_pred = np.zeros([3, 1])
    y_pred[0:2, 0] = x_pred[0:2, 0]
    y_pred[2, 0] = np.dot(np.transpose(x[2:4, :]), (r_ue - x_pred[0:2, :]))/np.linalg.norm((r_ue - x_pred[0:2, :])) + np.dot(np.transpose(x[2:4, :]), (r_bs - x_pred[0:2, :]))/np.linalg.norm(r_bs - x_pred[0:2, :])

    x_new = x_pred + np.dot(K, y - y_pred)
    P_new = np.dot(np.eye(4) - np.dot(K, H), P_pred)

    return x_new, P_new