import numpy as np
import torch
from sigKer_fast import sig_kernels_fb_mmd, sig_kernels_f_mmd

# MMD distance with gradients (fast)

class SigMMD(torch.nn.Module):

    def __init__(self, n=0):
        super(SigMMD, self).__init__()
        self.n = n

    def forward(self, X, Y):
        K_XX = SigKerGramMat.apply(X,None,self.n)
        K_YY = SigKerGramMat.apply(Y,None,self.n)
        K_XY = SigKerGramMat.apply(X,Y,self.n)
        MMD_squared = torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)
        return MMD_squared


class SigKerGramMat(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y=None, n=0):
        """
        input
         - X a list of A paths each of shape (M,D)
         - Y a list of B paths each of shape (N,D)

        computes by solving PDEs (forward) and by variation of parameter (backward)
         - (forward) K_XY: a Gram matrix of pairwise kernel evaluations k(x_i,y_j)   (A,B,M,D)  
         - (backward) K_dXY: Gram matrix of gradients ( dk(x_i,y_j)/dx_i )_{i,j} (A,B,M,D)   
        """
        XX, YY, XY = False, False, False

        if Y is None:
            Y = X.detach().clone() 
            if X.requires_grad:
                XX = True
            else:
                YY = True
        else:
            XY = True

        A = len(X)
        B = len(Y)
        D = X[0].shape[1]
        M = X[0].shape[0]

        # 1. FORWARD
        if XX:
            K, K_rev = sig_kernels_fb_mmd(X.detach().numpy(),Y.detach().numpy(),n,sym=True) 
            K_rev = torch.tensor(K_rev, dtype=torch.double)
        elif XY:
            K, K_rev =  sig_kernels_fb_mmd(X.detach().numpy(),Y.detach().numpy(),n,sym=False) 
            K_rev = torch.tensor(K_rev, dtype=torch.double)
        else:
            K = sig_kernels_f_mmd(Y.detach().numpy(),n) 
        K = torch.tensor(K, dtype=torch.double)
      
        # 2. GRADIENTS
        if XX or XY: 
            # Need to get the increments of Y on the finer grid
            inc_Y = (Y[:,1:,:]-Y[:,:-1,:])/float(2**n)  #(B,N-1,D)  increments defined by the data
            inc_Y = tile(inc_Y,1,2**n)                  #(B,(2**n)*(N-1),D)  increments on the finer grid

            # Need to reorganize the K_rev matrix
            K_rev_rev = flip(K_rev,dim=2)              # (A,B,(2**n)*(M-1),(2**n)*(N-1))
            K_rev_rev = flip(K_rev_rev,dim=3)          # (A,B,(2**n)*(M-1),(2**n)*(N-1))

            KK = (K[:,:,:-1,:-1] * K_rev_rev[:,:,1:,1:])                       # (A,B,(2**n)*(M-1),(2**n)*(N-1))

            K_grad = KK[:,:,:,:,None]*inc_Y[None,:,None,:,:]                   # (A,B,(2**n)*(M-1),(2**n)*(N-1),D)

            K_grad = (1./(2**n))*torch.sum(K_grad,axis=3)                      # (A,B,(2**n)*(M-1),D)

            K_grad =  torch.sum(K_grad.reshape(A,B,M-1,2**n,D),axis=3)         # (A,B, M-1,D)

            ctx.save_for_backward(K_grad)
        
        ctx.XX, ctx.YY, ctx.XY = XX, YY, XY

        return K[:,:,-1,-1]

    @staticmethod
    def backward(ctx, grad_output):

        """
        1. During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        2. grad_output contains dL/dK_XY 
        """

        XX, YY, XY = ctx.XX, ctx.YY, ctx.XY

        if XX or XY:
            # from gradients w.r.t. increments to gradients w.r.t. points of the time series.
            grad_incr, = ctx.saved_tensors
            A = grad_incr.shape[0]
            B = grad_incr.shape[1]
            D = grad_incr.shape[3]
            grad_points = -torch.cat([grad_incr,torch.zeros((A, B, 1, D)).type(torch.float64)], dim=2) + torch.cat([torch.zeros((A, B,1, D)).type(torch.float64), grad_incr], dim=2)

        if XX:
            grad = (grad_output[:,:,None,None]*grad_points + grad_output.t()[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None  
        if YY:
            return None, None, None
        if XY:
            # dL/dX = dL/dK dK/dX = (\sum_j dL/dKij dKij/dXi)_i = (\sum_j{ grad_out_ij * dKij/dXi})_i for any loss L
            grad = (grad_output[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None


# Naive implementation of MMD distance with gradients ontain via pytorch automatic differentiation (slow)

class SigMMD_naive(torch.nn.Module):

    def __init__(self, n=0):
        super(SigMMD_naive, self).__init__()
        self.n = n

    def forward(self, X, Y):
        K_XX = SigKerGramMat_naive(X,X,self.n)
        K_YY = SigKerGramMat_naive(Y,Y,self.n)    
        K_XY = SigKerGramMat_naive(X,Y,self.n)
        return torch.mean(K_XX)+torch.mean(K_YY)-2.*torch.mean(K_XY)


def SigKerGramMat_naive(X,Y,n=0):
    """
    input
     - X a list of A paths each of shape (M,D)
     - Y a list of B paths each of shape (N,D)

    computes by solving PDEs (forward)
     -  K_XY: a gram matrix of pairwise kernel evaluations k(x_i,y_j) 
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


# Various utils functions

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)