import numpy as np
from scipy import integrate
from scipy import interpolate
from tools import *
import os
import tempfile
import shutil
from joblib import Parallel, delayed
import multiprocessing
import iisignature

class SigKernel():
    """Signature Kernel via PDE solution"""

    def __init__(self, X, Y, n=0, rough=False):
        # input paths
        self.X = X
        self.Y = Y
        # number of discretization steps between two increments
        self.n = n
        # path and vector space sizes
        if rough:
            self.M = X.shape[1]
            self.D = X.shape[2]
            self.N = Y.shape[1]
        else:
            self.M, self.D = X.shape
            self.N = Y.shape[0]
        # arithmetic type
        self.dtype = np.float64
        # matrix of inner product of increments
        if rough:
            self.increments = self.rough_incrementsMatrix()
        else:
            self.increments = self.incrementsMatrix()
        # use the rough lift instead of the underlying paths
        self.rough = rough

    def initialize_solution_K(self):
        """Set up first iterator in the Picard iteration (=0 everywhere except at boundary)"""
        K = np.zeros(((2**self.n)*(self.M-1)+1, (2**self.n)*(self.N-1)+1), dtype=self.dtype)
        K[0, :] = self.dtype(1.)
        K[:, 0] = self.dtype(1.)
        return K

    def incrementsMatrix(self):
        """Computes the matrix of inner product of increments of the input paths.
           Returns the matrix {<dX_i, dY_j>}_{ij}
        """
        mat = np.zeros(((2**self.n)*(self.M-1)+1, (2**self.n)*(self.N-1)+1), dtype=self.dtype)
        for i in range(0, (2**self.n)*(self.M-1)):
            for j in range(0, (2**self.n)*(self.N-1)):
                ii = int(i/(2**self.n))
                jj = int(j/(2**self.n))
                inc_X = (self.X[ii+1, :] - self.X[ii, :])/float(2**self.n)
                inc_Y = (self.Y[jj+1, :] - self.Y[jj, :])/float(2**self.n)
                mat[i, j] = (inc_X * inc_Y).sum()
        return mat

    def rough_incrementsMatrix(self):
        """Computes the matrix of inner product of increments of the input rough paths.
           Returns the matrix {<dS(X)_i, dS(Y)_j>}_{ij}
        """
        mat = np.zeros((self.M, self.N), dtype=self.dtype)
        for i in range(0, self.M-1):
            for j in range(0, self.N-1):
                if (i==self.M-1): 
                    inc_X = self.X.signature(i,None)
                else:
                    inc_X = self.X.signature(i,i+2)
                if (j==self.N-1):
                    inc_Y = self.Y.signature(j,None)
                else:
                    inc_Y = self.Y.signature(j,j+2)
                mat[i, j] = (inc_X * inc_Y).sum()
        return mat

    def Finite_Differences(self, K):
        """Forward finite difference scheme. It computes the sig kernel over a grid 
           (2**self.n)*(M-1) x (2**self.n)*(N-1)"""
        for i in range(1, (2 ** self.n) * (self.M - 1) + 1):
            for j in range(1, (2 ** self.n) * (self.N - 1) + 1):
                K[i,j] = K[i,j-1] + K[i-1,j] + K[i-1,j-1]*self.increments[i-1,j-1] - K[i-1,j-1]
        return K[::(2**self.n),::(2**self.n)]

    def rough_Finite_Differences(self, K):
        """Forward finite difference scheme. It computes the rough sig kernel over a grid 
           (2**self.n)*(M-1) x (2**self.n)*(N-1)"""
        for i in range(1, self.M):
            for j in range(1, self.N):
                K[i,j] = K[i,j-1] + K[i-1,j] + K[i-1,j-1]*self.increments[i-1,j-1] - K[i-1,j-1]
        return K

    def Kernel(self):
        """Returns signature Kernel by numerical integration"""
        K = self.initialize_solution_K()
        if self.rough:
            return self.rough_Finite_Differences(K)
        return self.Finite_Differences(K)
    
    
    
    
def sig_kernel(x,y,n=0):
    return SigKernel(x,y,n).Kernel()[-1,-1]

def naive(x,y,d=2):
    return (iisignature.sig(x,d)*iisignature.sig(y,d)).sum()

def processing_train(i, x_train, n, dm):
    L = len(x_train)
    for j in range(i, L):
        dm[i,j] = sig_kernel(x_train[i], x_train[j], n)
    
def covariance_train(x_train, n=0):
    
    L = len(x_train)

    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    dm = np.memmap(filename, dtype=float, shape=(L, L), mode='w+')

    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=5)(delayed(processing_train)(i, x_train, n, dm) for i in range(L))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass

    dm_inv = dm.T.copy() 
    np.fill_diagonal(dm_inv, 0.)
    dm = dm + dm_inv
    
    return dm

def processing_test(j, x_train, x_test, n, dm):
    L = len(x_train)
    LL = len(x_test)
    for i in range(L):
        dm[j,i] = sig_kernel(x_test[j], x_train[i], n)

def covariance_test(x_train, x_test, n=0):
    
    L = len(x_train)
    LL = len(x_test)
    
    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    dm = np.memmap(filename, dtype=float, shape=(LL, L), mode='w+')

    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), max_nbytes=None, verbose=5)(delayed(processing_test)(j, x_train, x_test, n, dm) for j in range(LL))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass
    
    return dm
 