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
@cuda.jit
def compute_sigKernel_backward_cuda(M_inc_rev, len_x, len_y, n_anti_diagonals, M_sol):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc_rev: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
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
        i = len_x - I
        j = len_y - J

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            M_sol[block_id, i, j] = M_sol[block_id, i-1, j] + M_sol[block_id, i, j-1] + M_sol[block_id, i-1, j-1]*(M_inc_rev[block_id, i-1, j-1]-1.)

        # Wait for other threads in this block
        cuda.syncthreads()
# ----------------------------------------------------------------------------------------------------------------------

class SigKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y=None):
        """
        input
         - X : 3-tensor of shape (N, len, dim)
         - Y : 3-tensor of shape (N, len, dim)
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

        A = X.shape[0]
        M = X.shape[1]
        D = X.shape[2]

        # Compute increment matrix
        M_inc = torch.bmm(X[:,1:,:]-X[:,:-1,:], (Y[:,1:,:]-Y[:,:-1,:]).permute(0,2,1))

        dev = M_inc.device
        dtype = M_inc.dtype

        threads_per_block = M
        n_anti_diagonals = 2 * threads_per_block - 1

        # Prepare the tensor of output solutions to the PDE (forward)
        K = torch.zeros((A, M+1, M+1), device=dev, dtype=dtype) 
        K[:,0,:] = 1.
        K[:,:,0] = 1. 

        # Run the CUDA kernel to compute the forward signature kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_sigKernel_forward_cuda[A, threads_per_block](cuda.as_cuda_array(M_inc.detach()), 
                                                             M, M, n_anti_diagonals,
                                                             cuda.as_cuda_array(K))

        K = K[:,:-1,:-1]

        # 1. FORWARD
        if XX or XY:
            
            # Compute reversed increment matrix
            X_rev = X.detach()[:,::-1,:]
            Y_rev = Y.detach()[:,::-1,:]
            M_inc_rev = torch.bmm(X_rev[:,1:,:]-X_rev[:,:-1,:], 
                                 (Y_rev[:,1:,:]-Y_rev[:,:-1,:]).permute(0,2,1))

            # Prepare the tensor of output solutions to the PDE (forward)
            K_rev = torch.zeros((A, M+1, M+1), device=dev, dtype=dtype) 
            K_rev[:,-1,:] = 1.
            K_rev[:,:,-1] = 1. 

            # Compute signature kernel for reversed paths
            compute_sigKernel_backward_cuda[A, threads_per_block](cuda.as_cuda_array(M_inc_rev.detach()), 
                                                                 M, M, n_anti_diagonals,
                                                                 cuda.as_cuda_array(K_rev))

            K_rev = K_rev[:,1:,1:]

        # 2. GRADIENTS
        if XX or XY: 
            inc_Y = Y[:,1:,:]-Y[:,:-1,:]                 # (A,M-1,D)  increments defined by the data   
            KK = (K[:,:-1,:-1] * K_rev[:,1:,1:])         # (A,(2**n)*(M-1),(2**n)*(N-1))
            K_grad = KK[:,:,:,None]*inc_Y[:,None,:,:]    # (A,(2**n)*(M-1),(2**n)*(N-1),D)
            K_grad = torch.sum(K_grad,axis=2)            # (A,(2**n)*(M-1),D)
            ctx.save_for_backward(K_grad)
        
        ctx.XX, ctx.YY, ctx.XY = XX, YY, XY

        return K[:,-1,-1]

    @staticmethod
    def backward(ctx, grad_output):

        """
        During the forward pass, the gradients with respect to each increment in each dimension has been computed.
        Here we derive the gradients with respect to the points of the time series.
        """

        XX, YY, XY = ctx.XX, ctx.YY, ctx.XY
     
        if XX or XY:
            grad_incr , = ctx.saved_tensors

            A = grad_incr.shape[0]
            D = grad_incr.shape[2]
            grad_points = -torch.cat([grad_incr,torch.zeros((A, 1, D)).type(torch.float64).to(grad_incr.device)], dim=1) + torch.cat([torch.zeros((A, 1, D)).type(torch.float64).to(grad_incr.device), grad_incr], dim=1)

        if XX:
            # remark1: grad_points=\sum_a dKa/dX, whilst dL/dX = \sum_a grad_output[a]*dKa/dX
            # where dKa/dX is a tensor of shape (A,M,N) with zeros everywhere except for Ka[a,:,:].
            # we need to 'inject grad_output' in grad_points, it corresponds to do grad_output[a]*grad_points[a,:,:]
            # remark2: KXX is bilinear, and grad_points is the gradient with respect to the left variable -> we need to multiply by 2
            return 2.*grad_output[:,None,None]*grad_points, None
        if YY:
            # should never go here
            return None, None
        if XY:
            # see remark 1
            return grad_output[:,None,None]*grad_points, None




class SigLoss(torch.nn.Module):

    def __init__(self, n_chunks=2):
        super(SigLoss, self).__init__()
        self.n_chunks = n_chunks
        
    def sig_distance(self,x,y):
        d = torch.mean( SigKernel.apply(x,None)+ SigKernel.apply(y,None)- 2.*SigKernel.apply(x,y) )
        return d #+ torch.mean((x[:,0,:]-y[:,0,:])**2) #+ torch.mean(torch.abs(x[:,-1,:]-y[:,-1,:]))

    def forward(self, X, Y):

        assert X.requires_grad and not Y.requires_grad, "the first input should require grad, and not the second"

        if self.n_chunks==1:
            return self.sig_distance(X,Y)

        dist = torch.tensor(0., dtype=torch.float64)
        for k in range(2, self.n_chunks+1):
            X_chunks = torch.chunk(X, k, dim=1)
            Y_chunks = torch.chunk(Y, k, dim=1)
            for x1,x2,y1,y2 in zip(X_chunks[:-1], X_chunks[1:], Y_chunks[:-1], Y_chunks[1:]):
                dist += self.sig_distance(torch.cat([x1,x2],dim=1),torch.cat([y1,y2],dim=1))

        return dist