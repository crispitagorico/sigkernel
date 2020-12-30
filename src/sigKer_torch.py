import numpy as np
import torch
import torch.cuda
from numba import jit
from numba import cuda

from sigKer_fast import sig_kernel_batch, sig_kernel_batch_

# ===========================================================================================================
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

            inc = M_inc[block_id, i-1, j-1]

            # vanilla scheme
            M_sol[block_id, i, j] = M_sol[block_id, i-1, j] + M_sol[block_id, i, j-1] + M_sol[block_id, i-1, j-1]*(inc-1.)

            # explicit scheme
            #M_sol[block_id, i, j] = (M_sol[block_id, i-1, j]+M_sol[block_id, i, j-1])*(1.+0.5*inc+(1./12)*inc**2) - M_sol[block_id, i-1, j-1]*(1.-(1./12)*inc**2)

        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================

# ===========================================================================================================
class SigKernelCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        """
        input
         - X : 3-tensor of shape (batch, len, dim)
         - Y : 3-tensor of shape (batch, len, dim)
        """

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        # Compute increment matrix
        M_inc = torch.bmm(X[:,1:,:]-X[:,:-1,:], (Y[:,1:,:]-Y[:,:-1,:]).permute(0,2,1))

        threads_per_block = max(M,N)
        n_anti_diagonals = 2 * threads_per_block - 1

        # Prepare the tensor of output solutions to the PDE (forward)
        K = torch.zeros((A, M+1, N+1), device=M_inc.device, dtype=M_inc.dtype) 
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
# ===========================================================================================================


# ===========================================================================================================
class SigKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y=None, n=0, solver=0, method='variation_parameters'):
        """
        Compute Signature Kernel and its gradients using two possible backpropagation methods
        X,Y are two 3-tensors of shape (batch, len, dim)
        n is the discretisation of the grid where the PDE gets solved. The higher is n, the more accurate the final output, but slower the code
        solver=0 means a vanila finite difference scheme, solver=1 means an explicit scheme, solver=2 means an implicit scheme (deprecated)
        method='variation_parameters' or else solve directly another PDE for gradients.
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
        D = X.shape[2]
        M = X.shape[1]

        # 1. FORWARD
        if XX or XY:
            if method=='variation_parameters':
                K, K_rev = sig_kernel_batch(X.detach().cpu().numpy(),Y.detach().cpu().numpy(),n=n,solver=solver,gradients=True) 
            else:
                K, K_rev = sig_kernel_batch_(X.detach().cpu().numpy(),Y.detach().cpu().numpy(),n=n,solver=solver,gradients=True)

            K_rev = torch.tensor(K_rev, dtype=torch.double).to(X.device)
        else:
            if method=='variation_parameters':
                K =  sig_kernel_batch(X.detach().cpu().numpy(),Y.detach().cpu().numpy(),n=n,solver=solver,gradients=False)
            else:
                K =  sig_kernel_batch_(X.detach().cpu().numpy(),Y.detach().cpu().numpy(),n=n,solver=solver,gradients=False)
        K = torch.tensor(K, dtype=torch.double).to(X.device)

        # 2. GRADIENTS
        if XX or XY: 
            if method=='variation_parameters':
                # Need to get the increments of Y on the finer grid
                inc_Y = (Y[:,1:,:]-Y[:,:-1,:])/float(2**n)  #(A,N-1,D)  increments defined by the data
                inc_Y = tile(inc_Y,1,2**n)                  #(A,(2**n)*(M-1),D)  increments on the finer grid

                # Need to reorganize the K_rev matrix
                K_rev_rev = flip(K_rev,dim=1)
                K_rev_rev = flip(K_rev_rev,dim=2)

                KK = (K[:,:-1,:-1] * K_rev_rev[:,1:,1:])                       # (A,(2**n)*(M-1),(2**n)*(N-1))

                K_grad = KK[:,:,:,None]*inc_Y[:,None,:,:]                      # (A,(2**n)*(M-1),(2**n)*(N-1),D)

                K_grad = (1./(2**n))*torch.sum(K_grad,axis=2)                  # (A,(2**n)*(M-1),D)

                K_grad =  torch.sum(K_grad.reshape(A,M-1,2**n,D),axis=2)       # (A,M-1,D)

                ctx.save_for_backward(K_grad)
            else:
                ctx.save_for_backward(K_rev[:,:,:,-1,-1])
        
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
            return 2.*grad_output[:,None,None]*grad_points, None, None, None, None
        if YY:
            # should never go here
            return None, None, None, None, None
        if XY:
            # see remark 1
            return grad_output[:,None,None]*grad_points, None, None, None, None
# ===========================================================================================================

# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================