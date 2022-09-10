import numpy as np
import torch
import torch.cuda
from numba import cuda

from cython_backend import sig_kernel_batch_varpar, sig_kernel_Gram_varpar #, sig_kernel_derivative_batch
from .cuda_backend import compute_sig_kernel_batch_varpar_from_increments_cuda, compute_sig_kernel_derivative_batch_from_increments_cuda, compute_sig_kernel_Gram_mat_varpar_from_increments_cuda


# ===========================================================================================================
# Static kernels
# ===========================================================================================================
class LinearKernel():
    """Linear kernel k: R^d x R^d -> R"""

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        return torch.bmm(X, Y.permute(0,2,1))

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        return torch.einsum('ipk,jqk->ijpq', X, Y)


class RBFKernel():
    """RBF kernel k: R^d x R^d -> R"""

    def __init__(self, sigma):
        self.sigma = sigma

    def batch_kernel(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^i_t) of shape (batch, length_X, length_Y)
        """
        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.bmm(X, Y.permute(0,2,1))
        dist += torch.reshape(Xs,(A,M,1)) + torch.reshape(Ys,(A,1,N))
        return torch.exp(-dist/self.sigma)

    def Gram_matrix(self, X, Y):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        Xs = torch.sum(X**2, dim=2)
        Ys = torch.sum(Y**2, dim=2)
        dist = -2.*torch.einsum('ipk,jqk->ijpq', X, Y)
        dist += torch.reshape(Xs,(A,1,M,1)) + torch.reshape(Ys,(1,B,1,N))
        return torch.exp(-dist/self.sigma)
# ===========================================================================================================



# ===========================================================================================================
# Signature Kernel
# ===========================================================================================================
class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self,static_kernel, dyadic_order, _naive_solver=False):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def compute_kernel(self, X, Y, max_batch=100):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector k(X^i_T,Y^i_T) of shape (batch,)
        """
        batch = X.shape[0]
        if batch <= max_batch:
            K = _SigKernel.apply(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver)
        else:
            cutoff = int(batch/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.compute_kernel(X1, Y1, max_batch)
            K2 = self.compute_kernel(X2, Y2, max_batch)
            K = torch.cat((K1, K2), 0)
        return K


    def compute_kernel_and_derivative(self, X, Y, gamma, max_batch=100):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim),
                  - gamma: torch tensor of shape (batch, length_X, dim)
           Output:
                  - vector of shape (batch,) of kernel evaluations k_gamma(X^i_T,Y^i_T)
                  - vector of shape (batch,) of directional derivatives k_gamma(X^i_T,Y^i_T) wrt 1st variable
        """

        batch = X.shape[0]
        if batch <= max_batch:
            K, K_grad = k_kgrad(X, Y, gamma, self.dyadic_order, self.static_kernel)
        else:
            cutoff = int(batch/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            g1, g2 = gamma[:cutoff], gamma[cutoff:]
            K1, K_grad1 = self.compute_kernel_and_derivative(X1, Y1, g1, max_batch)
            K2, K_grad2 = self.compute_kernel_and_derivative(X2, Y2, g2, max_batch)
            K = torch.cat((K1, K2), 0)
            K_grad = torch.cat((K_grad1, K_grad2), 0)
        return K, K_grad


    def compute_Gram(self, X, Y, sym=False, max_batch=100):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """

        batch_X = X.shape[0]
        batch_Y = Y.shape[0]
        if batch_X <= max_batch and batch_Y <= max_batch:
            K = _SigKernelGram.apply(X, Y, self.static_kernel, self.dyadic_order, sym, self._naive_solver)
        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y/2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.compute_Gram(X, Y1, False, max_batch)
            K2 = self.compute_Gram(X, Y2, False, max_batch)
            K = torch.cat((K1, K2), 1)
        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X/2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self.compute_Gram(X1, Y, False, max_batch)
            K2 = self.compute_Gram(X2, Y, False, max_batch)
            K = torch.cat((K1, K2), 0)
        else:
            cutoff_X = int(batch_X/2)
            cutoff_Y = int(batch_Y/2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
            K11 = self.compute_Gram(X1, Y1, False, max_batch)
            K12 = self.compute_Gram(X1, Y2, False, max_batch)
            K21 = self.compute_Gram(X2, Y1, False, max_batch)
            K22 = self.compute_Gram(X2, Y2, False, max_batch)
            K_top = torch.cat((K11, K12), 1)
            K_bottom = torch.cat((K21, K22), 1)
            K = torch.cat((K_top, K_bottom), 0)
        return K

    def compute_distance(self, X, Y, max_batch=100):
        """Input: 
                  - X: torch tensor of shape (batch, length_X, dim),
                  - Y: torch tensor of shape (batch, length_Y, dim)
           Output: 
                  - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """
        
        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_kernel(X, X, max_batch)
        K_YY = self.compute_kernel(Y, Y, max_batch)
        K_XY = self.compute_kernel(X, Y, max_batch)

        return torch.mean(K_XX) + torch.mean(K_YY) - 2.*torch.mean(K_XY)

    def compute_scoring_rule(self, X, y, max_batch=100):
        """Input:
                  - X: torch tensor of shape (batch, length_X, dim),
                  - y: torch tensor of shape (1, length_Y, dim)
           Output:
                  - signature kernel scoring rule S(X,y) = E[k(X,X)] - 2E[k(X,y]
        """

        assert not y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        K_Xy = self.compute_Gram(X, y, sym=False, max_batch=max_batch)

        K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))

        return K_XX_m - 2. * torch.mean(K_Xy)

    def compute_expected_scoring_rule(self, X, Y, max_batch=100):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - signature kernel expected scoring rule S(X,Y) = E_Y[S(X,y)]
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        K_XY = self.compute_Gram(X, Y, sym=False, max_batch=max_batch)

        K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))

        return K_XX_m - 2.*torch.mean(K_XY)

    def compute_mmd(self, X, Y, max_batch=100):
        """Input: 
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output: 
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"

        K_XX = self.compute_Gram(X, X, sym=True, max_batch=max_batch)
        K_YY = self.compute_Gram(Y, Y, sym=True, max_batch=max_batch)
        K_XY = self.compute_Gram(X, Y, sym=False, max_batch=max_batch)

        K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
        K_YY_m = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

        return K_XX_m + K_YY_m - 2. * torch.mean(K_XY)


class _SigKernel(torch.autograd.Function):
    """Signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""
 
    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, _naive_solver=False):

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^i_t)
        G_static = static_kernel.batch_kernel(X,Y)
        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type in ['cuda']:

            assert max(MM+1,NN+1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'
            
            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            K[:,0,:] = 1.
            K[:,:,0] = 1. 

            # Compute the forward signature kernel
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K), _naive_solver)
            K = K[:,:-1,:-1]

        # if on CPU
        else:
            K = torch.tensor(sig_kernel_batch_varpar(G_static_.detach().numpy(), _naive_solver), dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X,Y,G_static,K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return K[:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):
    
        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
        G_static_ = tile(tile(G_static_,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=1),dim=2)

        # if on GPU
        if X.device.type in ['cuda']:

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM+2, NN+2), device=G_static_rev.device, dtype=G_static_rev.dtype) 
            K_rev[:,0,:] = 1.
            K_rev[:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM,NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            compute_sig_kernel_batch_varpar_from_increments_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                       MM+1, NN+1, n_anti_diagonals,
                                                                                       cuda.as_cuda_array(K_rev), _naive_solver)

            K_rev = K_rev[:,:-1,:-1]      

        # if on CPU
        else:
            K_rev = torch.tensor(sig_kernel_batch_varpar(G_static_rev.detach().numpy(), _naive_solver), dtype=G_static.dtype, device=G_static.device)

        K_rev = flip(flip(K_rev,dim=1),dim=2)
        KK = K[:,:-1,:-1] * K_rev[:,1:,1:]   
        
        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D, dtype=X.dtype, device=X.device)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.batch_kernel(Xh,Y) 
        G_h = G_h.reshape(A,M,D,N)
        G_h = G_h.permute(0,1,3,2) 

        Diff_1 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_1 =  tile( tile(Diff_1,1,2**dyadic_order)/float(2**dyadic_order),2, 2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,1:,1:,:] - G_h[:,1:,:-1,:] - (G_static[:,1:,1:])[:,:,:,None] + (G_static[:,1:,:-1])[:,:,:,None]
        Diff_2 += - G_h[:,:-1,1:,:] + G_h[:,:-1,:-1,:] + (G_static[:,:-1,1:])[:,:,:,None] - (G_static[:,:-1,:-1])[:,:,:,None]
        Diff_2 = tile(tile(Diff_2,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (KK[:,:,:,None] * Diff_1)/h
        grad_2 = (KK[:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1, axis=2)
        grad_1 = torch.sum(grad_1.reshape(A,M-1,2**dyadic_order,D),axis=2)
        grad_2 = torch.sum(grad_2, axis=2)
        grad_2 = torch.sum(grad_2.reshape(A,M-1,2**dyadic_order,D),axis=2)

        grad_prev = grad_1[:,:-1,:] + grad_2[:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:,1:,:]],dim=1)   # /
        grad_incr = grad_prev - grad_1[:,1:,:]
        grad_points = torch.cat([(grad_2[:,0,:]-grad_1[:,0,:])[:,None,:],grad_incr,grad_1[:,-1,:][:,None,:]],dim=1)

        if Y.requires_grad:
            if torch.equal(X, Y):
                grad_points*=2
            else:
                raise NotImplementedError('Should implement the gradients for the case where both sets of inputs are diffentiable but are different')

        return grad_output[:,None,None]*grad_points, None, None, None, None



class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, sym=False, _naive_solver=False):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)

        # computing dsdt k(X^i_s,Y^j_t)
        G_static = static_kernel.Gram_matrix(X,Y)
        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        # if on GPU
        if X.device.type in ['cuda']:

            assert max(MM,NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G[:,:,0,:] = 1.
            G[:,:,:,0] = 1. 

            # Run the CUDA kernel.
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G), _naive_solver)

            G = G[:,:,:-1,:-1]

        else:
            G = torch.tensor(sig_kernel_Gram_varpar(G_static_.detach().numpy(), sym, _naive_solver), dtype=G_static.dtype, device=G_static.device)

        ctx.save_for_backward(X,Y,G,G_static)      
        ctx.sym = sym
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return G[:,:,-1,-1]


    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G, G_static = ctx.saved_tensors
        sym = ctx.sym
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
        G_static_ = tile(tile(G_static_,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2**dyadic_order)*(M-1)
        NN = (2**dyadic_order)*(N-1)
            
        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
        G_static_rev = flip(flip(G_static_,dim=2),dim=3)

        # if on GPU
        if X.device.type in ['cuda']:

            # Prepare the tensor of output solutions to the PDE (backward)
            G_rev = torch.zeros((A, B, MM+2, NN+2), device=G_static.device, dtype=G_static.dtype) 
            G_rev[:,:,0,:] = 1.
            G_rev[:,:,:,0] = 1. 

            # cuda parameters
            threads_per_block = max(MM+1,NN+1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            blockspergrid = (A,B)
            compute_sig_kernel_Gram_mat_varpar_from_increments_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()), 
                                                                                                      MM+1, NN+1, n_anti_diagonals,
                                                                                                      cuda.as_cuda_array(G_rev), _naive_solver)

            G_rev = G_rev[:,:,:-1,:-1]

        # if on CPU
        else:
            G_rev = torch.tensor(sig_kernel_Gram_varpar(G_static_rev.detach().numpy(), sym, _naive_solver), dtype=G_static.dtype, device=G_static.device)

        G_rev = flip(flip(G_rev,dim=2),dim=3)
        GG = G[:,:,:-1,:-1] * G_rev[:,:,1:,1:]  # shape (A,B,MM,NN)   

        # finite difference step 
        h = 1e-9

        Xh = X[:,:,:,None] + h*torch.eye(D, dtype=X.dtype, device=X.device)[None,None,:]  
        Xh = Xh.permute(0,1,3,2)
        Xh = Xh.reshape(A,M*D,D)

        G_h = static_kernel.Gram_matrix(Xh,Y) 
        G_h = G_h.reshape(A,B,M,D,N)
        G_h = G_h.permute(0,1,2,4,3) # shape (A,B,M,N,D)

        Diff_1 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_1 =  tile(tile(Diff_1,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  
        Diff_2 = G_h[:,:,1:,1:,:] - G_h[:,:,1:,:-1,:] - (G_static[:,:,1:,1:])[:,:,:,:,None] + (G_static[:,:,1:,:-1])[:,:,:,:,None]
        Diff_2 += - G_h[:,:,:-1,1:,:] + G_h[:,:,:-1,:-1,:] + (G_static[:,:,:-1,1:])[:,:,:,:,None] - (G_static[:,:,:-1,:-1])[:,:,:,:,None]
        Diff_2 = tile(tile(Diff_2,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)  

        grad_1 = (GG[:,:,:,:,None] * Diff_1)/h     # shape (A,B,MM,NN,D) 
        grad_2 = (GG[:,:,:,:,None] * Diff_2)/h

        grad_1 = torch.sum(grad_1,axis=3)    # shape (A,B,MM,D) 
        grad_1 = torch.sum(grad_1.reshape(A,B,M-1,2**dyadic_order,D),axis=3)   # shape (A,B,M-1,D) 
        grad_2 = torch.sum(grad_2,axis=3)    # shape (A,B,MM,D) 
        grad_2 = torch.sum(grad_2.reshape(A,B,M-1,2**dyadic_order,D),axis=3)   # shape (A,B,M-1,D) 

        grad_prev = grad_1[:,:,:-1,:] + grad_2[:,:,1:,:]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_1[:,:,1:,:]], dim=2)   # /
        grad_incr = grad_prev - grad_1[:,:,1:,:]
        grad_points = torch.cat([(grad_2[:,:,0,:]-grad_1[:,:,0,:])[:,:,None,:],grad_incr,grad_1[:,:,-1,:][:,:,None,:]],dim=2)   # shape (A,B,M,D) 

        if Y.requires_grad:
            if torch.equal(X, Y):
#                 grad = (grad_output[:,:,None,None]*grad_points + grad_output.t()[:,:,None,None]*grad_points).sum(dim=1)
                grad = 2*(grad_output[:,:,None,None]*grad_points).sum(dim=1)
                return grad, None, None, None, None, None
            else:
                grad = 2*(grad_output[:,:,None,None]*grad_points).sum(dim=1)
                return grad, None, None, None, None, None
#                 raise NotImplementedError('Should implement the gradients for the case where the gram matrix is non symmetric and both sets of inputs are diffentiable')
        else:
            grad = (grad_output[:,:,None,None]*grad_points).sum(dim=1)
            return grad, None, None, None, None, None
# ===========================================================================================================



def k_kgrad(X, Y, gamma, dyadic_order, static_kernel):
    """Input:
              - X: torch tensor of shape (batch, length_X, dim),
              - Y: torch tensor of shape (batch, length_Y, dim),
              - gamma: torch tensor of shape (batch, length_X, dim)
       Output:
              - vector of shape (batch,) of directional derivatives k_gamma(X^i_T,Y^i_T) wrt 1st variable
    """

    A = X.shape[0]
    M = X.shape[1]
    N = Y.shape[1]
    D = X.shape[2]

    MM = (2 ** dyadic_order) * (M - 1)
    NN = (2 ** dyadic_order) * (N - 1)

    G_static = static_kernel.batch_kernel(X, Y)
    G_static_diff = static_kernel.batch_kernel(gamma, Y)

    G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
    G_static_diff_ = G_static_diff[:, 1:, 1:] + G_static_diff[:, :-1, :-1] - G_static_diff[:, 1:,:-1] - G_static_diff[:, :-1, 1:]

    G_static_ = tile(tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                     2 ** dyadic_order) / float(2 ** dyadic_order)
    G_static_diff_ = tile(tile(G_static_diff_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                          2 ** dyadic_order) / float(2 ** dyadic_order)

    # if on GPU
    if X.device.type in ['cuda']:

        assert max(MM + 1, NN + 1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

        # cuda parameters
        threads_per_block = max(MM + 1, NN + 1)
        n_anti_diagonals = 2 * threads_per_block - 1

        # Prepare the tensor of output solutions to the PDE
        K = torch.zeros((A, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
        K_diff = torch.zeros((A, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)

        K[:, 0, :] = 1.
        K[:, :, 0] = 1.

        K_diff[:, 0, :] = 0.
        K_diff[:, :, 0] = 0.

        # Compute the signature kernel and its derivative
        compute_sig_kernel_derivative_batch_from_increments_cuda[A, threads_per_block](
            cuda.as_cuda_array(G_static_.detach()),
            cuda.as_cuda_array(G_static_diff_.detach()),
            MM + 1, NN + 1, n_anti_diagonals,
            cuda.as_cuda_array(K), cuda.as_cuda_array(K_diff))

        K = K[:, :-1, :-1]
        K_diff = K_diff[:, :-1, :-1]

    # if on CPU
    else:
        K, K_diff = sig_kernel_derivative_batch(G_static_.detach().numpy(), G_static_diff_.detach().numpy())
        K = torch.tensor(K, dtype=G_static.dtype, device=G_static.device)
        K_diff = torch.tensor(K_diff, dtype=G_static.dtype, device=G_static.device)

    return K[:, -1, -1], K_diff[:, -1, -1]


# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)
# ===========================================================================================================



# ===========================================================================================================
# Hypothesis test functionality
# ===========================================================================================================
def c_alpha(m, alpha):
    return 4. * np.sqrt(-np.log(alpha) / m)

def hypothesis_test(y_pred, y_test, static_kernel, confidence_level=0.99, dyadic_order=0):
    """Statistical test based on MMD distance to determine if 
       two sets of paths come from the same distribution.
    """

    k_sig = SigKernel(static_kernel, dyadic_order)

    m = max(y_pred.shape[0], y_test.shape[0])
    
    TU = k_sig.compute_mmd(y_pred,y_test)  
  
    c = torch.tensor(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(f'Hypothesis rejected: distribution are not equal with {confidence_level*100}% confidence')
    else:
        print(f'Hypothesis accepted: distribution are equal with {confidence_level*100}% confidence')
# ===========================================================================================================








# ===========================================================================================================
# Deprecated implementation (just for testing)
# ===========================================================================================================
def SigKernel_naive(X, Y, static_kernel, dyadic_order=0, _naive_solver=False):

    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2**dyadic_order)*(M-1)
    NN = (2**dyadic_order)*(N-1)

    K_XY = torch.zeros((A, MM+1, NN+1), dtype=X.dtype, device=X.device)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^i_t)
    G_static = static_kernel.batch_kernel(X,Y)
    G_static = G_static[:,1:,1:] + G_static[:,:-1,:-1] - G_static[:,1:,:-1] - G_static[:,:-1,1:] 
    G_static = tile(tile(G_static,1,2**dyadic_order)/float(2**dyadic_order),2,2**dyadic_order)/float(2**dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:,i,j].clone()

            k_10 = K_XY[:, i + 1, j].clone()
            k_01 = K_XY[:, i, j + 1].clone()
            k_00 = K_XY[:, i, j].clone()

            if _naive_solver:
                K_XY[:, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            else:
                K_XY[:, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
                #K_XY[:, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)
            
    return K_XY[:, -1, -1]


class SigLoss_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigLoss_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self,X,Y):

        k_XX = SigKernel_naive(X,X,self.static_kernel,self.dyadic_order,self._naive_solver)
        k_YY = SigKernel_naive(Y,Y,self.static_kernel,self.dyadic_order,self._naive_solver)
        k_XY = SigKernel_naive(X,Y,self.static_kernel,self.dyadic_order,self._naive_solver)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2.*torch.mean(k_XY)


def SigKernelGramMat_naive(X,Y,static_kernel,dyadic_order=0,_naive_solver=False):

    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2**dyadic_order)*(M-1)
    NN = (2**dyadic_order)*(N-1)

    K_XY = torch.zeros((A,B, MM+1, NN+1), dtype=X.dtype, device=X.device)
    K_XY[:,:, 0, :] = 1.
    K_XY[:,:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^j_t)
    G_static = static_kernel.Gram_matrix(X,Y)
    G_static = G_static[:,:,1:,1:] + G_static[:,:,:-1,:-1] - G_static[:,:,1:,:-1] - G_static[:,:,:-1,1:] 
    G_static = tile(tile(G_static,2,2**dyadic_order)/float(2**dyadic_order),3,2**dyadic_order)/float(2**dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:,:,i,j].clone()

            k_10 = K_XY[:, :, i + 1, j].clone()
            k_01 = K_XY[:, :, i, j + 1].clone()
            k_00 = K_XY[:, :, i, j].clone()

            if _naive_solver:
                K_XY[:, :, i + 1, j + 1] = k_10 + k_01 + k_00*(increment-1.)
            else:
                K_XY[:, :, i + 1, j + 1] = (k_10 + k_01)*(1.+0.5*increment+(1./12)*increment**2) - k_00*(1.-(1./12)*increment**2)
                #K_XY[:, :, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)

    return K_XY[:,:, -1, -1]


class SigMMD_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigMMD_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self, X, Y):

        K_XX = SigKernelGramMat_naive(X,X,self.static_kernel,self.dyadic_order,self._naive_solver)
        K_YY = SigKernelGramMat_naive(Y,Y,self.static_kernel,self.dyadic_order,self._naive_solver)  
        K_XY = SigKernelGramMat_naive(X,Y,self.static_kernel,self.dyadic_order,self._naive_solver)
        
        K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
        K_YY_m = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

        return K_XX_m + K_YY_m - 2. * torch.mean(K_XY)

