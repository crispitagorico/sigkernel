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
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    mat = np.memmap(filename, dtype=np.float64, shape=(A, B, (2**n)*(M-1)+1, (2**n)*(N-1)+1), mode='w+')

    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=5)(delayed(processing)(i, X, Y, N, n, mat) for i in range(0, (2**n)*(M-1)))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass
    
    return mat


def Finite_Differences(K, X, Y, n, increments):
    A = len(X)
    B = len(Y)
    M = X[0].shape[0]
    N = Y[0].shape[0]

    for i in range(1, (2**n)*(M-1)+1):
        for j in range(1, (2**n) *(N-1)+1):
            K[:,:,i,j] = K[:,:,i,j-1] + K[:,:,i-1,j] + K[:,:,i-1,j-1]*increments[:,:,i-1,j-1] - K[:,:,i-1,j-1]
    
    return K[:, :, ::(2**n), ::(2**n)]


def Kernel(X,Y,n):
    """Returns signature Kernel by numerical integration"""
    K = initialize_solution_K(X,Y,n)
    increments = incrementsMatrix(X,Y,n)
    return Finite_Differences(K,X,Y,n,increments)
    
    

def multi_sig_kernel(X,Y,n=0):
    return Kernel(X,Y,n)[:,:,-1,-1].T

