import torch

def covariance(X,Y,n=0):
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K = torch.zeros((A, B, M, N)).type(torch.Tensor)
    K[:, :, 0, :] = 1.
    K[:, :, :, 0] = 1.

    for i in range(0, (2**n)*(M-1)):
        for j in range(0, (2**n)*(N-1)):
            ii = int(i/(2**n))
            jj = int(j/(2**n))
            inc_X = (X[:, ii+1, :] - X[:, ii, :])/float(2**n)
            inc_Y = (Y[:, jj+1, :] - Y[:, jj, :])/float(2**n)
            increment = torch.einsum('ik,jk->ij', inc_X, inc_Y)
            K[:,:,ii+1,jj+1] = K[:,:,ii+1,jj].clone() + K[:,:,ii,jj+1].clone() + K[:,:,ii,jj].clone()*increment.clone() - K[:,:,ii,jj].clone()
    
    return K[:,:,-1,-1].T


def RoughMMD(X_train, y_train):
    K1 = covariance(X_train, X_train)
    K2 = covariance(X_train, y_train)
    K3 = covariance(y_train, y_train)
    return (1./(K1.shape[0]**2))*K1.sum() - (2./(K1.shape[0]*K3.shape[0]))*K2.sum() + (1./(K3.shape[0]**2))*K3.sum()

