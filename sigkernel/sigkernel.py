import numpy as np
import torch
import torch.cuda
from numba import cuda
import gc
from tqdm import tqdm

from .static_kernels import *
from .cuda_backend import sigkernel_cuda, sigkernel_Gram_cuda, sigkernel_derivatives_Gram_cuda
from cython_backend import sigkernel_cython, sigkernel_Gram_cython, sigkernel_derivatives_Gram_cython


# ===========================================================================================================
# Signature Kernel
# ===========================================================================================================
class SigKernel():
    """Wrapper of the signature kernel k_sig(x,y) = <S(f(x)),S(f(y))> where k(x,y) = <f(x),f(y)> is a given static kernel"""

    def __init__(self, static_kernel, dyadic_order, _naive_solver=False, return_all=False):
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver
        self.return_all = return_all

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
            cutoff = int(batch / 2)
            X1, X2 = X[:cutoff], X[cutoff:]
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.compute_kernel(X1, Y1, max_batch)
            K2 = self.compute_kernel(X2, Y2, max_batch)
            K = torch.cat((K1, K2), 0)
        return K

    def compute_kernel_and_derivatives_Gram(self, X, Y, gamma, max_batch=100):
        """Input:
                  - X: torch tensor of shape (batch_x, length_X, dim),
                  - Y: torch tensor of shape (batch_y, length_Y, dim),
                  - gamma: torch tensor of shape (batch_x, length_X, dim)
           Output:
                  - vector of shape (batch_x,batch_y) of kernel evaluations k(X^i_T,Y^j_T)
                  - vector of shape (batch,batch_y) of directional derivatives k_gamma^i(X^i_T,Y^j_T)
                  - vector of shape (batch,batch_y) of second directional derivatives k_gamma^igamma^i(X^i_T,Y^j_T)
        """

        batch_X = X.shape[0]
        batch_Y = Y.shape[0]
        if batch_X <= max_batch and batch_Y <= max_batch:
            K, K_diff, K_diffdiff = k_kgrad(X, Y, gamma, self.dyadic_order, self.static_kernel)
        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y / 2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1, K_diff1, K_diffdiff1 = self.compute_kernel_and_derivatives_Gram(X, Y1, gamma, max_batch)
            K2, K_diff2, K_diffdiff2 = self.compute_kernel_and_derivatives_Gram(X, Y2, gamma, max_batch)
            K = torch.cat((K1, K2), 1)
            K_diff = torch.cat((K_diff1, K_diff2), 1)
            K_diffdiff = torch.cat((K_diffdiff1, K_diffdiff2), 1)
        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X / 2)
            X1, X2 = X[:cutoff], X[cutoff:]
            gamma1, gamma2 = gamma[:cutoff], gamma[cutoff:]
            K1, K_diff1, K_diffdiff1 = self.compute_kernel_and_derivatives_Gram(X1, Y, gamma1, max_batch)
            K2, K_diff2, K_diffdiff2 = self.compute_kernel_and_derivatives_Gram(X2, Y, gamma2, max_batch)
            K = torch.cat((K1, K2), 0)
            K_diff = torch.cat((K_diff1, K_diff2), 0)
            K_diffdiff = torch.cat((K_diffdiff1, K_diffdiff2), 0)
        else:
            cutoff_X = int(batch_X / 2)
            cutoff_Y = int(batch_Y / 2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]
            gamma1, gamma2 = gamma[:cutoff_X], gamma[cutoff_X:]
            K11, K_diff11, K_diffdiff11 = self.compute_kernel_and_derivatives_Gram(X1, Y1, gamma1, max_batch)
            K12, K_diff12, K_diffdiff12 = self.compute_kernel_and_derivatives_Gram(X1, Y2, gamma1, max_batch)
            K21, K_diff21, K_diffdiff21 = self.compute_kernel_and_derivatives_Gram(X2, Y1, gamma2, max_batch)
            K22, K_diff22, K_diffdiff22 = self.compute_kernel_and_derivatives_Gram(X2, Y2, gamma2, max_batch)

            K_top, K_diff_top, K_diffdiff_top = torch.cat((K11, K12), 1), torch.cat((K_diff11, K_diff12), 1), torch.cat(
                (K_diffdiff11, K_diffdiff12), 1)
            K_bottom, K_diff_bottom, K_diffdiff_bottom = torch.cat((K21, K22), 1), torch.cat((K_diff21, K_diff22),
                                                                                             1), torch.cat(
                (K_diffdiff21, K_diffdiff22), 1)
            K, K_diff, K_diffdiff = torch.cat((K_top, K_bottom), 0), torch.cat((K_diff_top, K_diff_bottom),
                                                                               0), torch.cat(
                (K_diffdiff_top, K_diffdiff_bottom), 0)
        return K, K_diff, K_diffdiff

    def compute_Gram(self, X, Y, sym=False, max_batch=100, recursive=True):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_T,Y^j_T) of shape (batch_X, batch_Y)
        """

        batch_X = X.shape[0]
        batch_Y = Y.shape[0]

        if recursive:
            if batch_X <= max_batch and batch_Y <= max_batch:
                K = _SigKernelGram.apply(X, Y, self.static_kernel, self.dyadic_order,
                                         sym, self._naive_solver, self.return_all)
            elif batch_X <= max_batch and batch_Y > max_batch:
                cutoff = int(batch_Y / 2)
                Y1, Y2 = Y[:cutoff], Y[cutoff:]
                K1 = self.compute_Gram(X, Y1, False, max_batch)
                K2 = self.compute_Gram(X, Y2, False, max_batch)
                K = torch.cat((K1, K2), 1)
                del K1, K2, Y1, Y2
            elif batch_X > max_batch and batch_Y <= max_batch:
                cutoff = int(batch_X / 2)
                X1, X2 = X[:cutoff], X[cutoff:]
                K1 = self.compute_Gram(X1, Y, False, max_batch)
                K2 = self.compute_Gram(X2, Y, False, max_batch)
                K = torch.cat((K1, K2), 0)
                del K1, K2, X1, X2
            else:
                cutoff_X = int(batch_X / 2)
                cutoff_Y = int(batch_Y / 2)
                X1, X2 = X[:cutoff_X], X[cutoff_X:]
                Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

                K11 = self.compute_Gram(X1, Y1, False, max_batch)
                K12 = self.compute_Gram(X1, Y2, False, max_batch)
                K_top = torch.cat((K11, K12), 1)
                del K11, K12
                torch.cuda.empty_cache()
                gc.collect()

                K21 = self.compute_Gram(X2, Y1, False, max_batch)
                K22 = self.compute_Gram(X2, Y2, False, max_batch)
                K_bottom = torch.cat((K21, K22), 1)
                del K21, K22
                torch.cuda.empty_cache()
                gc.collect()

                del X1, X2, Y1, Y2
                torch.cuda.empty_cache()
                gc.collect()

                K = torch.cat((K_top, K_bottom), 0)
                del K_top, K_bottom

        else:

            if self.return_all:
                len_X = X.shape[1]
                len_Y = Y.shape[1]

                K = torch.zeros((batch_X, batch_Y, len_X, len_Y), device=X.device, dtype=X.dtype)

            else:
                K = torch.zeros((batch_X, batch_Y), device=X.device, dtype=X.dtype)

            for i in tqdm(range(0, batch_X, max_batch)):
                for j in range(0, batch_Y, max_batch):

                    if i == j and i + max_batch < batch_X and j + max_batch < batch_Y:
                        temp = _SigKernelGram.apply(X[i:i + max_batch], Y[j:j + max_batch], self.static_kernel,
                                                    self.dyadic_order,
                                                    sym, self._naive_solver, self.return_all)
                    else:
                        temp = _SigKernelGram.apply(X[i:i + max_batch], Y[j:j + max_batch], self.static_kernel,
                                                    self.dyadic_order,
                                                    False, self._naive_solver, self.return_all)
                    K[i:i + max_batch, j:j + max_batch] = temp

            del temp

        torch.cuda.empty_cache()
        gc.collect()

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

        return torch.mean(K_XX) + torch.mean(K_YY) - 2. * torch.mean(K_XY)

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

        return K_XX_m - 2. * torch.mean(K_XY)

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

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X^i_s,Y^i_t)
        G_static = static_kernel.batch_kernel(X, Y)
        G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
        G_static_ = tile(tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        # if on GPU
        if X.device.type in ['cuda']:

            assert max(MM + 1,
                       NN + 1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM + 1, NN + 1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            K = torch.zeros((A, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
            K[:, 0, :] = 1.
            K[:, :, 0] = 1.

            # Compute the forward signature kernel
            sigkernel_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                 MM + 1, NN + 1, n_anti_diagonals,
                                                 cuda.as_cuda_array(K), _naive_solver)
            K = K[:, :-1, :-1]

        # if on CPU
        else:
            K = torch.tensor(sigkernel_cython(G_static_.detach().numpy(), _naive_solver), dtype=G_static.dtype,
                             device=G_static.device)

        ctx.save_for_backward(X, Y, G_static, K)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        ctx._naive_solver = _naive_solver

        return K  # [:,-1,-1]

    @staticmethod
    def backward(ctx, grad_output):

        X, Y, G_static, K = ctx.saved_tensors
        static_kernel = ctx.static_kernel
        dyadic_order = ctx.dyadic_order
        _naive_solver = ctx._naive_solver

        G_static_ = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
        G_static_ = tile(tile(G_static_, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        A = X.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # Reverse paths
        X_rev = torch.flip(X, dims=[1])
        Y_rev = torch.flip(Y, dims=[1])

        # computing dsdt k(X_rev^i_s,Y_rev^i_t) for variation of parameters
        G_static_rev = flip(flip(G_static_, dim=1), dim=2)

        # if on GPU
        if X.device.type in ['cuda']:

            # Prepare the tensor of output solutions to the PDE (backward)
            K_rev = torch.zeros((A, MM + 2, NN + 2), device=G_static_rev.device, dtype=G_static_rev.dtype)
            K_rev[:, 0, :] = 1.
            K_rev[:, :, 0] = 1.

            # cuda parameters
            threads_per_block = max(MM, NN)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Compute signature kernel for reversed paths
            sigkernel_cuda[A, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()),
                                                 MM + 1, NN + 1, n_anti_diagonals,
                                                 cuda.as_cuda_array(K_rev), _naive_solver)

            K_rev = K_rev[:, :-1, :-1]

        # if on CPU
        else:
            K_rev = torch.tensor(sigkernel_cython(G_static_rev.detach().numpy(), _naive_solver), dtype=G_static.dtype,
                                 device=G_static.device)

        K_rev = flip(flip(K_rev, dim=1), dim=2)
        KK = K[:, :-1, :-1] * K_rev[:, 1:, 1:]

        # finite difference step
        h = 1e-9

        Xh = X[:, :, :, None] + h * torch.eye(D, dtype=X.dtype, device=X.device)[None, None, :]
        Xh = Xh.permute(0, 1, 3, 2)
        Xh = Xh.reshape(A, M * D, D)

        G_h = static_kernel.batch_kernel(Xh, Y)
        G_h = G_h.reshape(A, M, D, N)
        G_h = G_h.permute(0, 1, 3, 2)

        Diff_1 = G_h[:, 1:, 1:, :] - G_h[:, 1:, :-1, :] - (G_static[:, 1:, 1:])[:, :, :, None] + (G_static[:, 1:, :-1])[
                                                                                                 :, :, :, None]
        Diff_1 = tile(tile(Diff_1, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
            2 ** dyadic_order)
        Diff_2 = G_h[:, 1:, 1:, :] - G_h[:, 1:, :-1, :] - (G_static[:, 1:, 1:])[:, :, :, None] + (G_static[:, 1:, :-1])[
                                                                                                 :, :, :, None]
        Diff_2 += - G_h[:, :-1, 1:, :] + G_h[:, :-1, :-1, :] + (G_static[:, :-1, 1:])[:, :, :, None] - (G_static[:, :-1,
                                                                                                        :-1])[:, :, :,
                                                                                                       None]
        Diff_2 = tile(tile(Diff_2, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
            2 ** dyadic_order)

        grad_1 = (KK[:, :, :, None] * Diff_1) / h
        grad_2 = (KK[:, :, :, None] * Diff_2) / h

        grad_1 = torch.sum(grad_1, axis=2)
        grad_1 = torch.sum(grad_1.reshape(A, M - 1, 2 ** dyadic_order, D), axis=2)
        grad_2 = torch.sum(grad_2, axis=2)
        grad_2 = torch.sum(grad_2.reshape(A, M - 1, 2 ** dyadic_order, D), axis=2)

        grad_prev = grad_1[:, :-1, :] + grad_2[:, 1:, :]  # /¯¯
        grad_next = torch.cat([torch.zeros((A, 1, D), dtype=X.dtype, device=X.device), grad_1[:, 1:, :]], dim=1)  # /
        grad_incr = grad_prev - grad_1[:, 1:, :]
        grad_points = torch.cat(
            [(grad_2[:, 0, :] - grad_1[:, 0, :])[:, None, :], grad_incr, grad_1[:, -1, :][:, None, :]], dim=1)

        return grad_output[:, None, None] * grad_points, None, None, None, None


class _SigKernelGram(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order, sym=False, _naive_solver=False, return_all=False):

        A = X.shape[0]
        B = Y.shape[0]
        M = X.shape[1]
        N = Y.shape[1]
        D = X.shape[2]

        MM = (2 ** dyadic_order) * (M - 1)
        NN = (2 ** dyadic_order) * (N - 1)

        # computing dsdt k(X^i_s,Y^j_t)
        G_static = static_kernel.Gram_matrix(X, Y)
        G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1,
                                                                                                  1:]
        G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3,
                         2 ** dyadic_order) / float(2 ** dyadic_order)

        # if on GPU
        if X.device.type in ['cuda']:

            assert max(MM,
                       NN) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

            # cuda parameters
            threads_per_block = max(MM + 1, NN + 1)
            n_anti_diagonals = 2 * threads_per_block - 1

            # Prepare the tensor of output solutions to the PDE (forward)
            G = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
            G[:, :, 0, :] = 1.
            G[:, :, :, 0] = 1.

            # Run the CUDA kernel.
            blockspergrid = (A, B)
            sigkernel_Gram_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                  MM + 1, NN + 1, n_anti_diagonals,
                                                                  cuda.as_cuda_array(G), _naive_solver)

            G = G[:, :, :-1, :-1]

        else:
            G = torch.tensor(sigkernel_Gram_cython(G_static_.detach().numpy(), sym, _naive_solver),
                             dtype=G_static.dtype, device=G_static.device)

        if X.requires_grad:
            grad_points = prep_backward(X, Y, G, G_static, sym, static_kernel, dyadic_order, _naive_solver)
            ctx.save_for_backward(X, Y, grad_points)

        if return_all:
            G_ = G[:, :, :: 2 ** dyadic_order, :: 2 ** dyadic_order]  # [:, :, -1, -1]
        else:
            G_ = G[:, :, -1, -1]

        del G
        torch.cuda.empty_cache()
        gc.collect()

        return G_

    @staticmethod
    def backward(ctx, grad_output):

        X, Y, grad_points = ctx.saved_tensors

        if Y.requires_grad:
            grad = 2 * (grad_output[:, :, None, None] * grad_points).sum(dim=1)
            return grad, None, None, None, None, None

        else:
            grad = (grad_output[:, :, None, None] * grad_points).sum(dim=1)
            return grad, None, None, None, None, None


# ===========================================================================================================

def prep_backward(X, Y, G, G_static, sym, static_kernel, dyadic_order, _naive_solver):
    G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1, 1:]
    G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
        2 ** dyadic_order)

    A = X.shape[0]
    B = Y.shape[0]
    M = X.shape[1]
    N = Y.shape[1]
    D = X.shape[2]

    MM = (2 ** dyadic_order) * (M - 1)
    NN = (2 ** dyadic_order) * (N - 1)

    # Reverse paths
    X_rev = torch.flip(X, dims=[1])
    Y_rev = torch.flip(Y, dims=[1])

    # computing dsdt k(X_rev^i_s,Y_rev^j_t) for variation of parameters
    G_static_rev = flip(flip(G_static_, dim=2), dim=3)

    # if on GPU
    if X.device.type in ['cuda']:

        # Prepare the tensor of output solutions to the PDE (backward)
        G_rev = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
        G_rev[:, :, 0, :] = 1.
        G_rev[:, :, :, 0] = 1.

        # cuda parameters
        threads_per_block = max(MM + 1, NN + 1)
        n_anti_diagonals = 2 * threads_per_block - 1

        # Compute signature kernel for reversed paths
        blockspergrid = (A, B)
        sigkernel_Gram_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_rev.detach()),
                                                              MM + 1, NN + 1, n_anti_diagonals,
                                                              cuda.as_cuda_array(G_rev), _naive_solver)

        G_rev = G_rev[:, :, :-1, :-1]

    # if on CPU
    else:
        G_rev = torch.tensor(sigkernel_Gram_cython(G_static_rev.detach().numpy(), sym, _naive_solver),
                             dtype=G_static.dtype, device=G_static.device)

    G_rev = flip(flip(G_rev, dim=2), dim=3)
    GG = G[:, :, :-1, :-1] * G_rev[:, :, 1:, 1:]  # shape (A,B,MM,NN)

    # finite difference step
    h = 1e-9

    Xh = X[:, :, :, None] + h * torch.eye(D, dtype=X.dtype, device=X.device)[None, None, :]
    Xh = Xh.permute(0, 1, 3, 2)
    Xh = Xh.reshape(A, M * D, D)

    G_h = static_kernel.Gram_matrix(Xh, Y)
    G_h = G_h.reshape(A, B, M, D, N)
    G_h = G_h.permute(0, 1, 2, 4, 3)  # shape (A,B,M,N,D)

    Diff_1 = G_h[:, :, 1:, 1:, :] - G_h[:, :, 1:, :-1, :] - (G_static[:, :, 1:, 1:])[:, :, :, :, None] + (G_static[:, :,
                                                                                                          1:, :-1])[:,
                                                                                                         :, :, :, None]
    Diff_1 = tile(tile(Diff_1, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
        2 ** dyadic_order)
    Diff_2 = G_h[:, :, 1:, 1:, :] - G_h[:, :, 1:, :-1, :] - (G_static[:, :, 1:, 1:])[:, :, :, :, None] + (G_static[:, :,
                                                                                                          1:, :-1])[:,
                                                                                                         :, :, :, None]
    Diff_2 += - G_h[:, :, :-1, 1:, :] + G_h[:, :, :-1, :-1, :] + (G_static[:, :, :-1, 1:])[:, :, :, :, None] - (
                                                                                                               G_static[
                                                                                                               :, :,
                                                                                                               :-1,
                                                                                                               :-1])[:,
                                                                                                               :, :, :,
                                                                                                               None]
    Diff_2 = tile(tile(Diff_2, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
        2 ** dyadic_order)

    grad_1 = (GG[:, :, :, :, None] * Diff_1) / h  # shape (A,B,MM,NN,D)
    grad_2 = (GG[:, :, :, :, None] * Diff_2) / h

    grad_1 = torch.sum(grad_1, axis=3)  # shape (A,B,MM,D)
    grad_1 = torch.sum(grad_1.reshape(A, B, M - 1, 2 ** dyadic_order, D), axis=3)  # shape (A,B,M-1,D)
    grad_2 = torch.sum(grad_2, axis=3)  # shape (A,B,MM,D)
    grad_2 = torch.sum(grad_2.reshape(A, B, M - 1, 2 ** dyadic_order, D), axis=3)  # shape (A,B,M-1,D)

    grad_prev = grad_1[:, :, :-1, :] + grad_2[:, :, 1:, :]  # /¯¯
    grad_next = torch.cat([torch.zeros((A, B, 1, D), dtype=X.dtype, device=X.device), grad_1[:, :, 1:, :]], dim=2)  # /
    grad_incr = grad_prev - grad_1[:, :, 1:, :]
    grad_points = torch.cat(
        [(grad_2[:, :, 0, :] - grad_1[:, :, 0, :])[:, :, None, :], grad_incr, grad_1[:, :, -1, :][:, :, None, :]],
        dim=2)  # shape (A,B,M,D)

    return grad_points


def k_kgrad(X, Y, gamma, dyadic_order, static_kernel):
    """Input:
              - X: torch tensor of shape (batch_x, length_X, dim),
              - Y: torch tensor of shape (batch_y, length_Y, dim),
              - gamma: torch tensor of shape (batch_x, length_X, dim)
       Output:
              - vector of shape (batch_x,batch_y) of signature kernel k(X^i_T,Y^j_T)
              - vector of shape (batch_x,batch_y) of directional derivatives k_gamma^i(X^i_T,Y^j_T)
              - vector of shape (batch_x,batch_y) of second directional derivatives k_{gamma^i gamma^i}(X^i_T,Y^j_T)
    """

    A = X.shape[0]
    B = Y.shape[0]
    M = X.shape[1]
    N = Y.shape[1]
    D = X.shape[2]

    MM = (2 ** dyadic_order) * (M - 1)
    NN = (2 ** dyadic_order) * (N - 1)

    G_static = static_kernel.Gram_matrix(X, Y)
    G_static_diff = static_kernel.Gram_matrix(gamma, Y)

    G_static_ = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1, 1:]
    G_static_diff_ = G_static_diff[:, :, 1:, 1:] + G_static_diff[:, :, :-1, :-1] - G_static_diff[:, :, 1:,
                                                                                   :-1] - G_static_diff[:, :, :-1, 1:]

    G_static_ = tile(tile(G_static_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
        2 ** dyadic_order)
    G_static_diff_ = tile(tile(G_static_diff_, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3,
                          2 ** dyadic_order) / float(2 ** dyadic_order)

    # if on GPU
    if X.device.type in ['cuda']:

        assert max(MM + 1,
                   NN + 1) < 1024, 'n must be lowered or data must be moved to CPU as the current choice of n makes exceed the thread limit'

        # cuda parameters
        threads_per_block = max(MM + 1, NN + 1)
        n_anti_diagonals = 2 * threads_per_block - 1

        # Prepare the tensor of output solutions to the PDE
        K = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
        K_diff = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)
        K_diffdiff = torch.zeros((A, B, MM + 2, NN + 2), device=G_static.device, dtype=G_static.dtype)

        K[:, :, 0, :] = 1.
        K[:, :, :, 0] = 1.

        # Compute the signature kernel and its derivative
        blockspergrid = (A, B)
        sigkernel_derivatives_Gram_cuda[blockspergrid, threads_per_block](cuda.as_cuda_array(G_static_.detach()),
                                                                          cuda.as_cuda_array(G_static_diff_.detach()),
                                                                          MM + 1, NN + 1, n_anti_diagonals,
                                                                          cuda.as_cuda_array(K),
                                                                          cuda.as_cuda_array(K_diff),
                                                                          cuda.as_cuda_array(K_diffdiff))

        K = K[:, :, :-1, :-1]
        K_diff = K_diff[:, :, :-1, :-1]
        K_diffdiff = K_diffdiff[:, :, :-1, :-1]

    # if on CPU
    else:
        K, K_diff, K_diffdiff = sigkernel_derivatives_Gram_cython(G_static_.detach().numpy(),
                                                                  G_static_diff_.detach().numpy())
        K = torch.tensor(K, dtype=G_static.dtype, device=G_static.device)
        K_diff = torch.tensor(K_diff, dtype=G_static.dtype, device=G_static.device)
        K_diffdiff = torch.tensor(K_diffdiff, dtype=G_static.dtype, device=G_static.device)

    return K[:, :, -1, -1], K_diff[:, :, -1, -1], K_diffdiff[:, :, -1, -1]


# ===========================================================================================================
# Various utility functions
# ===========================================================================================================
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


# ===========================================================================================================
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        a.device)
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

    TU = k_sig.compute_mmd(y_pred, y_test)

    c = torch.tensor(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(f'Hypothesis rejected: distribution are not equal with {confidence_level * 100}% confidence')
    else:
        print(f'Hypothesis accepted: distribution are equal with {confidence_level * 100}% confidence')


# ===========================================================================================================


def SigCHSIC(X, Y, Z, static_kernel, dyadic_order=1, eps=0.1):
    """Input:
            - X: torch tensor of shape (batch, length_X, dim_X),
            - Y: torch tensor of shape (batch, length_Y, dim_Y)
            - Z: torch tensor of shape (batch, length_Z, dim_Z)
            - sigma: kernel bandwidth hyperparameter
            - eps: normalization parameter

        Output: Signature CHSIC
    """

    device = X.device
    dtype = X.dtype

    # number of samples
    m = X.shape[0]

    # centering matrix
    H = torch.eye(m, dtype=dtype, device=device) - (1. / m) * torch.ones((m, m), dtype=dtype, device=device)

    # initialise signature kernel
    signature_kernel = SigKernel(static_kernel, dyadic_order)

    # compute signature Gram matrices
    K_X = signature_kernel.compute_Gram(X, X, sym=True)
    K_Y = signature_kernel.compute_Gram(Y, Y, sym=True)
    K_Z = signature_kernel.compute_Gram(Z, Z, sym=True)

    # center Gram matrices
    K_X_ = H @ K_X @ H
    K_Y_ = H @ K_Y @ H
    K_Z_ = H @ K_Z @ H

    # epsilon perturbation of K_Z_
    K_Z_e = K_Z_ + m * eps * torch.eye(m, device=device)

    # inverting K_Z_e
    K_Z_e_inv = torch.cholesky_inverse(K_Z_e)
    K_Z_e_inv2 = K_Z_e_inv @ K_Z_e_inv

    # computing three terms in CHSIC
    term_1 = torch.trace(K_X_ @ K_Y_)
    A = K_Z_ @ K_Z_e_inv2 @ K_Z_
    B = K_X_ @ A @ K_Y_
    term_2 = torch.trace(B)
    term_3 = torch.trace(B @ A)

    return (term_1 - 2. * term_2 + term_3) / m ** 2


# ===========================================================================================================
# Deprecated implementation (just for testing)
# ===========================================================================================================
def SigKernel_naive(X, Y, static_kernel, dyadic_order=0, _naive_solver=False):
    A = len(X)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2 ** dyadic_order) * (M - 1)
    NN = (2 ** dyadic_order) * (N - 1)

    K_XY = torch.zeros((A, MM + 1, NN + 1), dtype=X.dtype, device=X.device)
    K_XY[:, 0, :] = 1.
    K_XY[:, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^i_t)
    G_static = static_kernel.batch_kernel(X, Y)
    G_static = G_static[:, 1:, 1:] + G_static[:, :-1, :-1] - G_static[:, 1:, :-1] - G_static[:, :-1, 1:]
    G_static = tile(tile(G_static, 1, 2 ** dyadic_order) / float(2 ** dyadic_order), 2, 2 ** dyadic_order) / float(
        2 ** dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:, i, j].clone()

            k_10 = K_XY[:, i + 1, j].clone()
            k_01 = K_XY[:, i, j + 1].clone()
            k_00 = K_XY[:, i, j].clone()

            if _naive_solver:
                K_XY[:, i + 1, j + 1] = k_10 + k_01 + k_00 * (increment - 1.)
            else:
                K_XY[:, i + 1, j + 1] = (k_10 + k_01) * (1. + 0.5 * increment + (1. / 12) * increment ** 2) - k_00 * (
                            1. - (1. / 12) * increment ** 2)
                # K_XY[:, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)

    return K_XY[:, -1, -1]


class SigLoss_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigLoss_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self, X, Y):
        k_XX = SigKernel_naive(X, X, self.static_kernel, self.dyadic_order, self._naive_solver)
        k_YY = SigKernel_naive(Y, Y, self.static_kernel, self.dyadic_order, self._naive_solver)
        k_XY = SigKernel_naive(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver)

        return torch.mean(k_XX) + torch.mean(k_YY) - 2. * torch.mean(k_XY)


def SigKernelGramMat_naive(X, Y, static_kernel, dyadic_order=0, _naive_solver=False):
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    MM = (2 ** dyadic_order) * (M - 1)
    NN = (2 ** dyadic_order) * (N - 1)

    K_XY = torch.zeros((A, B, MM + 1, NN + 1), dtype=X.dtype, device=X.device)
    K_XY[:, :, 0, :] = 1.
    K_XY[:, :, :, 0] = 1.

    # computing dsdt k(X^i_s,Y^j_t)
    G_static = static_kernel.Gram_matrix(X, Y)
    G_static = G_static[:, :, 1:, 1:] + G_static[:, :, :-1, :-1] - G_static[:, :, 1:, :-1] - G_static[:, :, :-1, 1:]
    G_static = tile(tile(G_static, 2, 2 ** dyadic_order) / float(2 ** dyadic_order), 3, 2 ** dyadic_order) / float(
        2 ** dyadic_order)

    for i in range(MM):
        for j in range(NN):

            increment = G_static[:, :, i, j].clone()

            k_10 = K_XY[:, :, i + 1, j].clone()
            k_01 = K_XY[:, :, i, j + 1].clone()
            k_00 = K_XY[:, :, i, j].clone()

            if _naive_solver:
                K_XY[:, :, i + 1, j + 1] = k_10 + k_01 + k_00 * (increment - 1.)
            else:
                K_XY[:, :, i + 1, j + 1] = (k_10 + k_01) * (
                            1. + 0.5 * increment + (1. / 12) * increment ** 2) - k_00 * (
                                                       1. - (1. / 12) * increment ** 2)
                # K_XY[:, :, i + 1, j + 1] = k_01 + k_10 - k_00 + (torch.exp(0.5*increment) - 1.)*(k_01 + k_10)

    return K_XY[:, :, -1, -1]


class SigMMD_naive(torch.nn.Module):

    def __init__(self, static_kernel, dyadic_order=0, _naive_solver=False):
        super(SigMMD_naive, self).__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order
        self._naive_solver = _naive_solver

    def forward(self, X, Y):
        K_XX = SigKernelGramMat_naive(X, X, self.static_kernel, self.dyadic_order, self._naive_solver)
        K_YY = SigKernelGramMat_naive(Y, Y, self.static_kernel, self.dyadic_order, self._naive_solver)
        K_XY = SigKernelGramMat_naive(X, Y, self.static_kernel, self.dyadic_order, self._naive_solver)

        K_XX_m = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / (K_XX.shape[0] * (K_XX.shape[0] - 1.))
        K_YY_m = (torch.sum(K_YY) - torch.sum(torch.diag(K_YY))) / (K_YY.shape[0] * (K_YY.shape[0] - 1.))

        return K_XX_m + K_YY_m - 2. * torch.mean(K_XY)
