import torch

def SigKernel_AD(X,Y,n=0):
    """
    input
     - X a list of A paths each of shape (M,D)
     - Y a list of A paths each of shape (N,D)

    computes by solving PDEs (forward)
     -  K_XY: a vector of A pairwise kernel evaluations k(x_i,y_i)      (forward)
    """
    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K_XY = torch.zeros((A, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    for i in range(0, (2**n)*(M-1)):
        for j in range(0, (2**n)*(N-1)):

            ii = int(i / (2 ** n))
            jj = int(j / (2 ** n))

            inc_X_i = (X[:, ii + 1, :] - X[:, ii, :])/float(2**n)  # (A,D)
            inc_Y_j = (Y[:, jj + 1, :] - Y[:, jj, :])/float(2**n)  # (A,D)

            increment_XY = torch.einsum('ik,ik->i', inc_X_i, inc_Y_j)  # (A) <-> A dots prod bwn R^D and R^D

            K_XY[:, i + 1, j + 1] = K_XY[:, i + 1, j].clone() + K_XY[:, i, j + 1].clone() + K_XY[:, i,j].clone()* increment_XY.clone() - K_XY[ :, i,j].clone()

    return torch.mean(K_XY[:, -1, -1])
