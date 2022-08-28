# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp
import numpy as np

def sig_kernel_batch_varpar(double[:,:,:] G_static, bint _naive_solver=False):

    cdef int A = G_static.shape[0]
    cdef int M = G_static.shape[1]
    cdef int N = G_static.shape[2]
    cdef int i, j, l

    cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)

    for l in range(A):
        for i in range(M+1):
            K[l,i,0] = 1.

        for j in range(N+1):
            K[l,0,j] = 1.

        for i in range(M):
            for j in range(N):

                if _naive_solver:
                    K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1. + 0.5*G_static[l,i,j]) - K[l,i,j]
                    #K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] + K[l,i,j]*(G_static[l,i,j] - 1.)
                else:
                    K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1.+0.5*G_static[l,i,j]+(1./12)*G_static[l,i,j]**2) - K[l,i,j]*(1. - (1./12)*G_static[l,i,j]**2)
                    #K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] - K[l,i,j] + (exp(0.5*G_static[l,i,j])-1.)*(K[l,i+1,j] + K[l,i,j+1])

    return np.array(K)


def sig_kernel_derivative_batch(double[:,:,:] G_static, double[:,:,:] G_static_direction):

    cdef int A = G_static.shape[0]
    cdef int M = G_static.shape[1]
    cdef int N = G_static.shape[2]
    cdef int i, j, l

    cdef double[:,:,:] K_diff = np.zeros((A,M+1,N+1), dtype=np.float64)
    cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)

    for l in range(A):

        for i in range(M+1):
            K[l,i,0] = 1.
            K_diff[l,i,0] = 0.

        for j in range(N+1):
            K[l,0,j] = 1.
            K_diff[l,0,j] = 0.

        for i in range(M):
            for j in range(N):
                #K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1. + 0.5*G_static[l,i,j]) - K[l,i,j]
                K[l,i+1,j+1] = (K[l,i,j+1] + K[l,i+1,j]) * (1. + 0.5 * G_static[l,i,j] + (1. / 12) * G_static[l,i,j] ** 2) - K[l,i,j] * (1. - (1. / 12) * G_static[l,i,j] ** 2)
                K_diff[l,i+1,j+1] = (K_diff[l,i+1,j] + K_diff[l,i,j+1])*(1. + 0.5*G_static[l,i,j]) - K_diff[l,i,j] + (K[l,i+1,j] + K[l,i,j+1])*0.5*G_static[l,i,j]

    return np.array(K), np.array(K_diff)


def sig_kernel_Gram_varpar(double[:,:,:,:] G_static, bint sym=False, bint _naive_solver=False):

    cdef int A = G_static.shape[0]
    cdef int B = G_static.shape[1]
    cdef int M = G_static.shape[2]
    cdef int N = G_static.shape[3]
    cdef int i, j, l, m

    cdef double[:,:,:,:] K = np.zeros((A,B,M+1,N+1), dtype=np.float64)

    if sym:
        # for l in prange(A,nogil=True):
        for l in range(A):
            for m in range(l,A):

                for i in range(M+1):
                    K[l,m,i,0] = 1.
                    K[m,l,i,0] = 1.

                for j in range(N+1):
                    K[l,m,0,j] = 1.
                    K[m,l,0,j] = 1.

                for i in range(M):
                    for j in range(N):

                        if _naive_solver:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]) - K[l,m,i,j]
                            #K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j]-1.)
                        else:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1.+0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1.-(1./12)*G_static[l,m,i,j]**2)
                            #K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])

                        K[m,l,j+1,i+1] = K[l,m,i+1,j+1]

    else:
        # for l in prange(A,nogil=True):
        for l in range(A):
            for m in range(B):

                for i in range(M+1):
                    K[l,m,i,0] = 1.

                for j in range(N+1):
                    K[l,m,0,j] = 1.

                for i in range(M):
                    for j in range(N):

                        if _naive_solver:
                            K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j] - 1.)
                        else:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1. - (1./12)*G_static[l,m,i,j]**2)
                            #K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])

    return np.array(K)
