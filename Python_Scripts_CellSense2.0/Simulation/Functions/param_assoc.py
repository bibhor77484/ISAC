import numpy as np


### Permutations of path indices for different values of existing paths and new paths to be allocated. 
# For example, permute3_2 gives all the permutations of 2 paths allocated to 3 existing paths. 
## List of all possible permutations
permute1_3 = np.array([[0,6,6],
                       [6,0,6],
                       [6,6,0]])

permute2_3 = np.array([[0,1,6],
                      [1,0,6],
                      [0,6,1],
                      [1,6,0],
                      [6,0,1],
                      [6,1,0]])

permute3_3 = np.array([[0,1,2],
                     [0,2,1],
                     [1,0,2], 
                     [1,2,0], 
                     [2,0,1], 
                     [2,1,0]])

permute4_3 = np.array([[0,1,2],
                     [0,2,1],
                     [1,0,2], 
                     [1,2,0], 
                     [2,0,1], 
                     [2,1,0], 
                     [3,1,2],
                     [3,2,1],
                     [3,0,2], 
                     [3,2,0], 
                     [3,0,1], 
                     [3,1,0], 
                     [0,3,2],
                     [0,3,1],
                     [1,3,2], 
                     [1,3,0], 
                     [2,3,1], 
                     [2,3,0], 
                     [0,1,3],
                     [0,2,3],
                     [1,0,3], 
                     [1,2,3], 
                     [2,0,3], 
                     [2,1,3]])

permute5_3 = np.array([[0,1,2],
                     [0,2,1],
                     [1,0,2],
                     [1,2,0],
                     [2,0,1],
                     [2,1,0],
                     [3,1,2],
                     [3,2,1],
                     [3,0,2],
                     [3,2,0],
                     [3,0,1],
                     [3,1,0],
                     [0,3,2],
                     [0,3,1],
                     [1,3,2],
                     [1,3,0],
                     [2,3,1],
                     [2,3,0],
                     [0,1,3],
                     [0,2,3],
                     [1,0,3],
                     [1,2,3],
                     [2,0,3],
                     [2,1,3],
                     [4,1,2],
                     [4,2,1],
                     [4,0,2],
                     [4,2,0],
                     [4,0,1],
                     [4,1,0],
                     [0,4,2],
                     [0,4,1],
                     [1,4,2],
                     [1,4,0],
                     [2,4,1],
                     [2,4,0],
                     [0,1,4],
                     [0,2,4],
                     [1,0,4],
                     [1,2,4],
                     [2,0,4],
                     [2,1,4],
                     [4,1,3],
                     [4,2,3],
                     [4,0,3],
                     [4,3,0],
                     [4,3,1],
                     [4,3,2],
                     [0,4,3],
                     [1,4,3],
                     [2,4,3],
                     [3,4,0],
                     [3,4,1],
                     [3,4,2],
                     [0,3,4],
                     [1,3,4],
                     [2,3,4],
                     [3,0,4],
                     [3,1,4],
                     [3,2,4]])

permute1_2 = np.array([[0,6],
                       [6,0]])

permute2_2 = np.array([[0,1],
                       [1,0]])

permute3_2 = np.array([[0,1],
                       [0,2],
                       [1,0],
                       [1,2],
                       [2,0],
                       [2,1]])

permute4_2 = np.array([[0,1],
                       [0,2],
                       [0,3],
                       [1,0],
                       [1,2],
                       [1,3],
                       [2,0],
                       [2,1],
                       [2,3],
                       [3,0],
                       [3,1],
                       [3,2]])

permute5_2 = np.array([[0,1],
                      [0,2],
                      [0,3],
                      [0,4],
                      [1,0],
                      [1,2],
                      [1,3],
                      [1,4],
                      [2,0],
                      [2,1],
                      [2,3],
                      [2,4],
                      [3,0],
                      [3,1],
                      [3,2],
                      [3,4],
                      [4,0],
                      [4,1],
                      [4,2],
                      [4,3]])


# Returns the optimal assignment of estimated parameters to the parameters repository based on minimum loss.
def param_assoc(track_params_repo, t_est, theta_est, l_idx):

    # Why does greedy not work here? - Because when doing greedy we may assign an outlier path to a track_params index which should have been assigned to the next 
    # path. And we won't be able to assign the next path properly. 
    # We are implementing brute force method by searching through all possible permutations because the number of paths is very small (3 to 5). 
    # For larger number of paths, we can implement Hungarian algorithm or any other efficient algorithm for assignment problem.
    
    loss = np.zeros([7, len(t_est)])
    loss[0:l_idx, 0:len(t_est)] = np.abs(np.tile(np.reshape(t_est, [1, len(t_est)]), [l_idx, 1]) - np.tile(track_params_repo[0:l_idx, 0:1], [1, len(t_est)])) + \
        np.abs(np.tile(np.reshape(theta_est, [1, len(t_est)]), [l_idx, 1]) - np.tile(track_params_repo[0:l_idx, 1:2], [1, len(t_est)]))

    if (l_idx == 1) and (len(t_est) == 3):  permute = permute1_3
    if (l_idx == 2) and (len(t_est) == 3):  permute = permute2_3
    if (l_idx == 3) and (len(t_est) == 3):  permute = permute3_3
    if (l_idx == 4) and (len(t_est) == 3):  permute = permute4_3
    if (l_idx == 5) and (len(t_est) == 3):  permute = permute5_3
    if (l_idx == 1) and (len(t_est) == 2):  permute = permute1_2
    if (l_idx == 2) and (len(t_est) == 2):  permute = permute2_2
    if (l_idx == 3) and (len(t_est) == 2):  permute = permute3_2
    if (l_idx == 4) and (len(t_est) == 2):  permute = permute4_2
    if (l_idx == 5) and (len(t_est) == 2):  permute = permute5_2
    if len(t_est) == 1:                     permute = np.reshape(np.arange(0, l_idx, 1), [l_idx, 1])

    min_loss = 10000
    min_idx = -1

    for iter1 in range(0, np.shape(permute)[0]):
        if len(t_est) == 3:
            indices = ((permute[iter1, 0], permute[iter1, 1], permute[iter1, 2]), (0, 1, 2))
        if len(t_est) == 2:
            indices = ((permute[iter1, 0], permute[iter1, 1]), (0, 1))
        if len(t_est) == 1:
            indices = (permute[iter1, 0], 0)
        
        permuted_loss = np.sum(loss[indices])
        
        if permuted_loss < min_loss:
            min_loss = permuted_loss
            min_idx = iter1
        
    asgnmnt = permute[min_idx, :]

    return asgnmnt, loss