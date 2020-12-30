import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_sigKernel_forward_cuda(M_inc, len_x, len_y, n_anti_diagonals, M_sol):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            M_sol[block_id, i, j] = M_sol[block_id, i-1, j] + M_sol[block_id, i, j-1] + M_sol[block_id, i-1, j-1]*(M_inc[block_id, i-1, j-1]-1.)

        # Wait for other threads in this block
        cuda.syncthreads()

#def forward_step_explicit(double k_00, double k_01, double k_10, double increment):
#	return (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)

#def forward_step_implicit(double k_00, double k_01, double k_10, double increment):
#	return k_01+k_10-k_00 + ((0.5*increment)/(1.-0.25*increment))*(k_01+k_10)

# ----------------------------------------------------------------------------------------------------------------------

class SigKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        """
        input
         - X : 3-tensor of shape (N, len, dim)
         - Y : 3-tensor of shape (N, len, dim)
        """

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        # Compute increment matrix
        M_inc = torch.bmm(X[:,1:,:]-X[:,:-1,:], (Y[:,1:,:]-Y[:,:-1,:]).permute(0,2,1))

        dev = M_inc.device
        dtype = M_inc.dtype

        threads_per_block = max(M,N)
        n_anti_diagonals = 2 * threads_per_block - 1

        # Prepare the tensor of output solutions to the PDE (forward)
        K = torch.zeros((A, M+1, N+1), device=dev, dtype=dtype) 
        K[:,0,:] = 1.
        K[:,:,0] = 1. 

        # Run the CUDA kernel to compute the forward signature kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_sigKernel_forward_cuda[A, threads_per_block](cuda.as_cuda_array(M_inc.detach()), 
                                                             M, N, n_anti_diagonals,
                                                             cuda.as_cuda_array(K))

        K = K[:,:-1,:-1]

        ctx.save_for_backward(X,Y,K)

        return K[:,-1,-1]

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """
    
        X, Y, K = ctx.saved_tensors
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]
            
        # Compute reversed increment matrix
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])
        M_inc_rev = torch.bmm(X_rev[:,1:,:]-X_rev[:,:-1,:], (Y_rev[:,1:,:]-Y_rev[:,:-1,:]).permute(0,2,1))

        # Prepare the tensor of output solutions to the PDE (forward)
        K_rev = torch.zeros((A, M+1, N+1), device=dev, dtype=dtype) 
        K_rev[:,0,:] = 1.
        K_rev[:,:,0] = 1. 

        # Compute signature kernel for reversed paths
        compute_sigKernel_forward_cuda[A, threads_per_block](cuda.as_cuda_array(M_inc_rev.detach()), 
                                                             M, N, n_anti_diagonals,
                                                             cuda.as_cuda_array(K_rev))

        K_rev = K_rev[:,:-1,:-1]

        inc_Y = Y[:,1:,:]-Y[:,:-1,:]                       # (A,M-1,D)  increments defined by the data   
        
        K_rev = flip(flip(K_rev,dim=1),dim=2)
        
        KK = (K[:,:-1,:-1] * K_rev[:,1:,1:])               # (A,(2**n)*(M-1),(2**n)*(N-1))
        
        grad_incr = KK[:,:,:,None]*inc_Y[:,None,:,:]       # (A,(2**n)*(M-1),(2**n)*(N-1),D)
        
        grad_incr = torch.sum(grad_incr,axis=2)            # (A,(2**n)*(M-1),D)
        
        if Y.requires_grad:
            grad_incr*=2


        grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(grad_incr.dtype).to(grad_incr.device)], dim=1) + torch.cat([torch.zeros((A, 1, D)).type(grad_incr.dtype).to(grad_incr.device), grad_incr], dim=1)

        return grad_output[:,None,None]*grad_points, None


class SigLoss(torch.nn.Module):

    def __init__(self, n_chunks=2):
        super(SigLoss, self).__init__()
        self.n_chunks = n_chunks
        
    def sig_distance(self,x,y):
        d = torch.mean( SigKernel.apply(x,x)+ SigKernel.apply(y,y)- 2.*SigKernel.apply(x,y) )
        return d #+ torch.mean((x[:,0,:]-y[:,0,:])**2) #+ torch.mean(torch.abs(x[:,-1,:]-y[:,-1,:]))

    def forward(self, X, Y):

        assert not Y.requires_grad, "the second input should not require grad"

        if self.n_chunks==1:
            return self.sig_distance(X,Y)

        dist = torch.tensor(0., dtype=torch.float64)
        for k in range(2, self.n_chunks+1):
            X_chunks = torch.chunk(X, k, dim=1)
            Y_chunks = torch.chunk(Y, k, dim=1)
            for x1,x2,y1,y2 in zip(X_chunks[:-1], X_chunks[1:], Y_chunks[:-1], Y_chunks[1:]):
                dist += self.sig_distance(torch.cat([x1,x2],dim=1),torch.cat([y1,y2],dim=1))

        return dist


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
