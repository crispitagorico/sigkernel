import numpy as np
from tools import *
import iisignature

import os
import tempfile
import shutil
from joblib import Parallel, delayed
import multiprocessing


def initialize_solution_K(X,Y,n):
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    K = np.zeros((A, B, (2**n)*(M-1)+1, (2**n)*(N-1)+1), dtype=np.float64)
    K[:, :, 0, :] = 1.
    K[:, :, :, 0] = 1.
    
    return K


def processing(i,X,Y,N,n,mat):
    for j in range(0, (2**n)*(N-1)):
        ii = int(i/(2**n))
        jj = int(j/(2**n))
        inc_X = (X[:, ii+1, :] - X[:, ii, :])/float(2**n)
        inc_Y = (Y[:, jj+1, :] - Y[:, jj, :])/float(2**n)
        mat[:, :, i, j] = np.einsum('ik,jk->ij', inc_X, inc_Y)


def incrementsMatrix(X,Y,n):
    """Computes the matrix of inner product of increments of the input paths.
       Returns the matrix {<dX_i, dY_j>}_{ij}"""
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    mat = np.memmap(filename, dtype=np.float64, shape=(A, B, (2**n)*(M-1)+1, (2**n)*(N-1)+1), mode='w+')

    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=0)(delayed(processing)(i, X, Y, N, n, mat) for i in range(0, (2**n)*(M-1)))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass
    
    return mat


def Finite_Differences(K, X, Y, n, increments):
    """Forward finite difference scheme. It computes the sig kernel over a grid 
      (2**self.n)*(M-1) x (2**self.n)*(N-1)"""
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    for i in range(1, (2**n)*(M-1)+1):
        for j in range(1, (2**n) *(N-1)+1):
            K[:,:,i,j] = K[:,:,i,j-1] + K[:,:,i-1,j] + K[:,:,i-1,j-1]*increments[:,:,i-1,j-1] - K[:,:,i-1,j-1]
    
    return K[:, :, ::(2**n), ::(2**n)]


def KernelTensor(X,Y,n):
    """Returns signature Kernel by numerical integration
       
    input: X : np.array (nb_items_X, length_X, dim)
           Y : np.array (nb_items_Y, length_Y, dim)
    for two list of paths X and Y returns a (nb_items_X, nb_items_Y, length_X, length_Y)
    tensor K where K[i,j,m,n] is the sig kernel of the path X[i] with Y[j] up to time m and n respectively
    """
    K = initialize_solution_K(X,Y,n)
    increments = incrementsMatrix(X,Y,n)
    return Finite_Differences(K,X,Y,n,increments)


def sig_kernel_all_times(x, y, n=0):
    """Computes full signature kernel (for all times) between two paths x, y solving a Goursat problem
       
       input: x (np.array) (length_x, dim)
              y (np.array) (length_y, dim)
              n (int) discretization grid (default n=0)
       returns a (length_x, length_y) array K=K{i,j} where K[i,j] = <S(x)_i, S(y)_j>
    """
    return KernelTensor(np.array([x]), np.array([y]), n)[0,0,:,:]

def sig_kernel(x, y, n=0):
    """Computes full signature kernel between two paths x, y solving a Goursat problem
       
       input: x (np.array) (length, dim)
              y (np.array) (length, dim)
              n (int) discretization grid (default n=0)
       returns the float <S(x), S(y)>
    """
    return KernelTensor(np.array([x]), np.array([y]), n)[0,0,-1,-1]


def sig_covariance(X,Y,n=0):
    """input: two arrays of paths
       X : np.array (nb_items_X, length_X, dim)
       Y : np.array (nb_items_Y, length_Y, dim)
       returns the full signature kernel covariance matrix (nb_items_X, nb_items_Y)
       of the sets of paths X and Y"""

    return KernelTensor(X,Y,n)[:,:,-1,-1].T
