# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

def sig_kernel(double[:,:] x, double[:,:] y, int n=0):
	
	cdef int D = x.shape[1]
	cdef int M = x.shape[0]
	cdef int N = y.shape[0]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:] K = np.zeros((MM+1,NN+1), dtype=np.float64)

	for i in range(MM+1):
		K[i,0] = 1.

	for j in range(NN+1):
		K[0,j] = 1.

	for i in range(MM):
		for j in range(NN):

			ii = int(i/(2**n))
			jj = int(j/(2**n))

			increment = 0.
			for k in range(D):
				increment += (x[ii+1,k]-x[ii,k])*(y[jj+1,k]-y[jj,k])/factor

			K[i+1,j+1] = K[i,j+1] + K[i+1,j] + increment*K[i,j] - K[i,j]

	return K[MM,NN]


def sig_distance(double[:,:] x, double[:,:] y, int n=0):
	cdef double a = sig_kernel(x,x,n)
	cdef double b = sig_kernel(y,y,n)
	cdef double c = sig_kernel(x,y,n)
	return a + b - 2.*c


def sig_kernels_mmd(double[:,:,:] x, double[:,:,:] y, int n=0, bint sym=False):

	cdef int A = x.shape[0]
	cdef int B = y.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:,:] K = np.zeros((A,B,MM+1,NN+1), dtype=np.float64)

	if sym:
		for l in range(A):
			for m in range(l,A):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
					K[m,l,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.
					K[m,l,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
						K[l,m,i+1,j+1] = K[l,m,i,j+1] + K[l,m,i+1,j] + (increment-1.)*K[l,m,i,j]
						K[m,l,i+1,j+1] = K[l,m,i+1,j+1]


	else:
		for l in range(A):
			for m in range(B):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
	
						K[l,m,i+1,j+1] = K[l,m,i,j+1] + K[l,m,i+1,j] + (increment-1.)*K[l,m,i,j]
	
	return K