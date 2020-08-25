import torch

class SigLoss_torch_mmd(torch.nn.Module):

    def __init__(self, n=0):
        super(SigLoss_torch_mmd, self).__init__()
        self.n = n

    def forward(self, X, Y):
        K_XX = SigKernel_torch_gram(X,X,self.n)
        K_YY = SigKernel_torch_gram(Y,Y,self.n)    
        K_XY = SigKernel_torch_gram(X,Y,self.n)
        return torch.mean(K_XX)+torch.mean(K_YY)-2.*torch.mean(K_XY)


def SigKernel_torch_gram(X,Y,n=0):
    """
    input
     - X a list of A paths each of shape (M,D)
     - Y a list of B paths each of shape (N,D)

    computes by solving PDEs (forward)
     -  K_XY: a gram matrix of pairwise kernel evaluations k(x_i,y_j)      (forward)
    """
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K_XY = torch.zeros((A,B, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
    K_XY[:,:, 0, :] = 1.
    K_XY[:,:, :, 0] = 1.

    for i in range(0, (2**n)*(M-1)):
        for j in range(0, (2**n)*(N-1)):

            ii = int(i / (2 ** n))
            jj = int(j / (2 ** n))

            inc_X_i = (X[:, ii + 1, :] - X[:,ii, :])/float(2**n)  # (A,D)
            inc_Y_j = (Y[:, jj + 1, :] - Y[:, jj, :])/float(2**n)  # (B,D)

            increment_XY = torch.einsum('ik,jk->ij', inc_X_i, inc_Y_j)  # (A,B) 

            K_XY[:,:, i + 1, j + 1] = K_XY[:,:, i + 1, j].clone() + K_XY[:,:, i, j + 1].clone() + K_XY[:,:, i,j].clone()* increment_XY.clone() - K_XY[ :,:, i,j].clone()

    return K_XY[:,:, -1, -1]
