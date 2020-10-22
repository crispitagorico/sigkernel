# cython: boundscheck=False
# cython: wraparound=False

import numpy as np

def sig_kernels_fb_batch(double[:,:,:] x, double[:,:,:] y, int n=0):

	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment, increment_rev
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:] K = np.zeros((A,MM+1,NN+1), dtype=np.float64)
	cdef double[:,:,:] K_rev = np.zeros((A,MM+1,NN+1), dtype=np.float64)

	for l in range(A):

		for i in range(MM+1):
			K[l,i,0] = 1.
			K_rev[l,i,0] = 1.
	
		for j in range(NN+1):
			K[l,0,j] = 1.
			K_rev[l,0,j] = 1.

		for i in range(MM):
			for j in range(NN):

				ii = int(i/(2**n))
				jj = int(j/(2**n))

				increment = 0.
				increment_rev = 0.
				for k in range(D):
					increment += (x[l,ii+1,k]-x[l,ii,k])*(y[l,jj+1,k]-y[l,jj,k])/factor
					increment_rev += (x[l,(M-1)-(ii+1),k]-x[l,(M-1)-ii,k])*(y[l,(N-1)-(jj+1),k]-y[l,(N-1)-jj,k])/factor

				K[l,i+1,j+1] = K[l,i,j+1] + K[l,i+1,j] + (increment-1.)*K[l,i,j]
				K_rev[l,i+1,j+1] = K_rev[l,i,j+1] + K_rev[l,i+1,j] + (increment_rev-1.)*K_rev[l,i,j]

	return K, K_rev


def sig_kernels_f_batch(double[:,:,:] x, double[:,:,:] y, int n=0):

	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:] K = np.zeros((A,MM+1,NN+1), dtype=np.float64)

	for l in range(A):

		for i in range(MM+1):
			K[l,i,0] = 1.
	
		for j in range(NN+1):
			K[l,0,j] = 1.

		for i in range(MM):
			for j in range(NN):

				ii = int(i/(2**n))
				jj = int(j/(2**n))

				increment = 0.
				increment_inv = 0.
				for k in range(D):
					increment += (x[l,ii+1,k]-x[l,ii,k])*(y[l,jj+1,k]-y[l,jj,k])/factor

				K[l,i+1,j+1] = K[l,i,j+1] + K[l,i+1,j] + (increment-1.)*K[l,i,j]

	return K


def sig_kernels_fb_mmd(double[:,:,:] x, double[:,:,:] y, int n=0, bint sym=False):

	cdef int A = x.shape[0]
	cdef int B = y.shape[0]
	cdef int M = x.shape[1]
	cdef int N = y.shape[1]
	cdef int D = x.shape[2]

	cdef double increment, increment_rev
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)
	cdef int NN = (2**n)*(N-1)

	cdef double[:,:,:,:] K = np.zeros((A,B,MM+1,NN+1), dtype=np.float64)
	cdef double[:,:,:,:] K_rev = np.zeros((A,B,MM+1,NN+1), dtype=np.float64)

	if sym:
		for l in range(A):
			for m in range(l,A):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
					K_rev[l,m,i,0] = 1.
					K[m,l,i,0] = 1.
					K_rev[m,l,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.
					K_rev[l,m,0,j] = 1.
					K[m,l,0,j] = 1.
					K_rev[m,l,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						increment_rev = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
							increment_rev += (x[l,(M-1)-(ii+1),k]-x[l,(M-1)-ii,k])*(y[m,(N-1)-(jj+1),k]-y[m,(N-1)-jj,k])/factor

						K[l,m,i+1,j+1] = K[l,m,i,j+1] + K[l,m,i+1,j] + (increment-1.)*K[l,m,i,j]
						K[m,l,i+1,j+1] = K[l,m,i+1,j+1]
						K_rev[l,m,i+1,j+1] = K_rev[l,m,i,j+1] + K_rev[l,m,i+1,j] + (increment_rev-1.)*K_rev[l,m,i,j]
						K_rev[m,l,i+1,j+1]=K_rev[l,m,i+1,j+1]

	else:
		for l in range(A):
			for m in range(B):

				for i in range(MM+1):
					K[l,m,i,0] = 1.
					K_rev[l,m,i,0] = 1.
	
				for j in range(NN+1):
					K[l,m,0,j] = 1.
					K_rev[l,m,0,j] = 1.

				for i in range(MM):
					for j in range(NN):

						ii = int(i/(2**n))
						jj = int(j/(2**n))

						increment = 0.
						increment_rev = 0.
						for k in range(D):
							increment += (x[l,ii+1,k]-x[l,ii,k])*(y[m,jj+1,k]-y[m,jj,k])/factor
							increment_rev += (x[l,(M-1)-(ii+1),k]-x[l,(M-1)-ii,k])*(y[m,(N-1)-(jj+1),k]-y[m,(N-1)-jj,k])/factor

						K[l,m,i+1,j+1] = K[l,m,i,j+1] + K[l,m,i+1,j] + (increment-1.)*K[l,m,i,j]
						K_rev[l,m,i+1,j+1] = K_rev[l,m,i,j+1] + K_rev[l,m,i+1,j] + (increment_rev-1.)*K_rev[l,m,i,j]

	return K, K_rev


def sig_kernels_f_mmd(double[:,:,:] x, int n=0):

	cdef int A = x.shape[0]
	cdef int M = x.shape[1]
	cdef int D = x.shape[2]

	cdef double increment
	cdef double factor = 2**(2*n)

	cdef int i, j, k, l, ii, jj
	cdef int MM = (2**n)*(M-1)

	cdef double[:,:,:,:] K = np.zeros((A,A,MM+1,MM+1), dtype=np.float64)

	for l in range(A):
		for m in range(l,A):

			for i in range(MM+1):
				K[l,m,i,0] = 1.
				K[m,l,i,0] = 1.
	
			for j in range(MM+1):
				K[l,m,0,j] = 1.
				K[m,l,0,j] = 1.

			for i in range(MM):
				for j in range(MM):

					ii = int(i/(2**n))
					jj = int(j/(2**n))

					increment = 0.

					for k in range(D):
						increment += (x[l,ii+1,k]-x[l,ii,k])*(x[m,jj+1,k]-x[m,jj,k])/factor
		
					K[l,m,i+1,j+1] = K[l,m,i,j+1] + K[l,m,i+1,j] + (increment-1.)*K[l,m,i,j]
					K[m,l,i+1,j+1] = K[l,m,i+1,j+1]
	return K