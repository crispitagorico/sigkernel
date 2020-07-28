import torch

class SigKernel_PDEs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y,n=0):
        """
        input
         - X a list of A paths each of shape (M,D)
         - Y a list of A paths each of shape (N,D)

        computes by solving PDEs
         -  K_XY: a vector of A pairwise kernel evaluations k(x_i,y_i)      (forward)
         -  K_dXY: A matrices ( dk_{x_{pq}}(x_i,y_i) )_{p=1,q=1}^{p=M,q=D}  (backward)

        """

        A = len(X)
        D = X[0].shape[1]
        M = X[0].shape[0]
        N = Y[0].shape[0]

        K_XY = torch.zeros((A, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
        K_XY[:, 0, :] = 1.
        K_XY[:, :, 0] = 1.

        # for the backward pass
        K_dXY = torch.zeros((A, M-1, D, (2**n)*(M-1)+1, (2**n)*(N-1)+1)).type(torch.float64)
        K_dXY[:, :, :, 0, :] = 0.  # A*M*D PDEs on MxN grid
        K_dXY[:, :, :, :, 0] = 0.  # to check


        for i in range(0, (2**n)*(M-1)):
            for j in range(0, (2**n)*(N-1)):

                ii = int(i / (2 ** n))
                jj = int(j / (2 ** n))

                inc_X_i = (X[:, ii + 1, :] - X[:, ii, :])/float(2**n)  # (A,D)
                inc_Y_j = (Y[:, jj + 1, :] - Y[:, jj, :])/float(2**n)  # (A,D) (1,D)

                increment_XY = torch.einsum('ik,ik->i', inc_X_i, inc_Y_j)  # (A) <-> A dots prod bwn R^D and R^D
                K_XY[:, i + 1, j + 1] = K_XY[:, i + 1, j] + K_XY[:, i, j + 1] + K_XY[:, i,j]* increment_XY - K_XY[ :, i,j]

                increment_HY = torch.zeros((A, M-1, D)).type(torch.float64)
                increment_HY[:, ii, :] = inc_Y_j  # (A,N-1,D) only for D PDE we have a non-zero dot product
                K_dXY[:, :, :, i + 1, j + 1] = K_dXY[:, :, :, i + 1, j] + K_dXY[:, :, :, i, j + 1] + K_dXY[:, :,:, i, j] * increment_XY[:,None,None] - K_dXY[:,:,:,i,j] + K_XY[:,i,j][:,None,None]*increment_HY

        ctx.save_for_backward(K_dXY[:, :, :, -1, -1])

        return torch.mean(K_XY[:, -1, -1])

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """

        grad_incr, = ctx.saved_tensors

        A = grad_incr.shape[0]
        D = grad_incr.shape[2]
        grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(torch.float64)], dim=1)+ torch.cat([torch.zeros((A, 1, D)).type(torch.float64), grad_incr], dim=1)

        return grad_points, None, None