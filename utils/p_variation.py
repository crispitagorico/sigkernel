#!/bin/env python3

import math
import collections

# for tests
import random
import time

def p_var_backbone(path_size, p, path_dist):
    # Input:
    # * path_size >= 0 integer
    # * p >= 1 real
    # * path_dist: metric on the set {0,...,path_dist-1}.
    #   Namely, path_dist(a,b) needs to be defined and nonnegative
    #   for all integer 0 <= a,b < path_dist, be symmetric and
    #   satisfy the triangle inequality:
    #   * path_dist(a,b) = path_dist(b,a)
    #   * path_dist(a,b) + path_dist(b,c) >= path_dist(a,c)
    #   Indiscernibility is not necessary, so path_dist may not
    #   be a metric in the strict sense.
    # Output: a class with two fields:
    # * .p_var = max sum_k path_dist(a_{k-1}, a_k)^p
    #            over all strictly increasing subsequences a_k of 0,...,path_size-1
    # * .points = the maximising sequence a_k
    # Notes:
    # * if path_size == 0, the result is .p_var = -math.inf, .points = []
    # * if path_size == 1, the result is .p_var = 0,         .points = [0]

    ret = collections.namedtuple('p_var', ['value', 'points'])

    if path_size == 0:
        return ret(value = -math.inf, points = [])
    elif path_size == 1:
        return ret(value = 0, points = [0])

    s = path_size - 1
    N = 1
    while s >> N != 0:
        N += 1

    ind = [0.0] * s
    def ind_n(j, n):
        return (s >> n) + (j >> n)
    def ind_k(j, n):
        return min(((j >> n) << n) + (1 << (n-1)), s);

    max_p_var = 0.0
    run_p_var = [0.0] * path_size

    point_links = [0] * path_size

    for j in range(0, path_size):
        for n in range(1, N + 1):
            if not(j >> n == s >> n and (s >> (n-1)) % 2 == 0):
                ind[ind_n(j, n)] = max(ind[ind_n(j, n)], path_dist(ind_k(j, n), j))
        if j == 0:
            continue

        m = j - 1
        delta = 0.0
        delta_m = j
        n = 0
        while True:
            while n > 0 and m >> n == s >> n and (s >> (n-1)) % 2 == 0:
                n -= 1;

            skip = False
            if n > 0:
                iid = ind[ind_n(m, n)] + path_dist(ind_k(m, n), j)
                if delta >= iid:
                    skip = True
                elif m < delta_m:
                    delta = pow(max_p_var - run_p_var[m], 1. / p)
                    delta_m = m
                    if delta >= iid:
                        skip = True

            if skip:
                k = (m >> n) << n
                if k > 0:
                    m = k - 1
                    while n < N and (k >> n) % 2 == 0:
                        n += 1
                else:
                    break
            else:
                if n > 1:
                    n -= 1
                else:
                    d = path_dist(m, j)
                    if d >= delta:
                        new_p_var = run_p_var[m] + pow(d, p)
                        if new_p_var >= max_p_var:
                            max_p_var = new_p_var
                            point_links[j] = m
                    if m > 0:
                        while n < N and (m >> n) % 2 == 0:
                            n += 1
                        m -= 1
                    else:
                        break
        run_p_var[j] = max_p_var

    points = []
    point_i = s
    while True:
        points.append(point_i)
        if point_i == 0:
            break
        point_i = point_links[point_i]
    points.reverse()
    return ret(value = run_p_var[-1], points = points)

def p_var_backbone_ref(path_size, p, path_dist):
    # Reference implementation of p_var_backbone, does not need the triangle inequality
    # but may be slow; obviously correct.
    if path_size == 0:
        return -math.inf
    elif path_size == 1:
        return 0
    cum_p_var = [0.0] * path_size
    for j in range(1, path_size):
        for m in range(0, j):
            cum_p_var[j] = max(cum_p_var[j], cum_p_var[m] + pow(path_dist(m, j), p));
    return cum_p_var[-1]

def p_var_points_check(p_var_ret, p, path_dist):
    # Check the output of p_var_backbone: whether the p-variation p_var_ret.value
    # is indeed reached on the sequence p_var_ret.points.
    # Return abs value of the error.

    if len(p_var_ret.points) == 0:
        if p_var_ret.value == -math.inf:
            return 0
        else:
            return math.inf

    if len(p_var_ret.points) == 1:
        return abs(p_var_ret.value)

    v = 0.0
    for k in range(1, len(p_var_ret.points)):
        v += pow(path_dist(p_var_ret.points[k-1], p_var_ret.points[k]), p)
    return abs(v - p_var_ret.value)

def ex_sq():
    # Example: unit square
    path = [[0,0], [0,1], [1,1], [1,0], [0,0], [0,0], [0,0], [0,0]]
    dist = lambda a, b: math.sqrt(pow(path[b][0] - path[a][0], 2) + pow(path[b][1] - path[a][1], 2))

    print(f'\nSquare path: {path}\nwith L^2 distance')
    p = 1.0
    while p <= 4.0:
        pv = p_var_backbone(len(path), p, dist)
        pv_ref = p_var_backbone_ref(len(path), p, dist)
        pv_err = abs(pv.value - pv_ref) + p_var_points_check(pv, p, dist)
        print(f'{p:5.2f}-variation: {pv.value:7.2f}, error {pv_err:.2e}, sequence {pv.points}')
        p += 0.5

def ex_bm():
    # Example: Brownian motion made of iid -1/+1 increments
    n = 2500
    print(f'\nPoor man\'s Brownian path with {n} steps:')
    path = [0.0] * (n + 1)
    sigma = 1. / math.sqrt(n)
    for k in range(1, n + 1):
        path[k] = path[k - 1] + random.choice([-1, 1]) * sigma
    dist = lambda a, b: abs(path[b] - path[a])

    for p in [1.0, math.sqrt(2), 2.0, math.exp(1)]:
        pv_start = time.time()
        pv = p_var_backbone(len(path), p, dist)
        pv_time = time.time() - pv_start
        pv_ref_start = time.time()
        pv_ref = p_var_backbone_ref(len(path), p, dist)
        pv_ref_time = time.time() - pv_ref_start
        pv_err = abs(pv.value - pv_ref) + p_var_points_check(pv, p, dist)
        print(f'{p:5.2f}-variation: {pv.value:7.2f}, sequence length: {len(pv.points):5d}, error {pv_err:.2e}, time: {pv_time:7.2f}, reference time: {pv_ref_time:7.2f}')

def ex_bm_long():
    # Example: long Brownian motion made of iid -1/+1 increments, no error check
    print('\nVery poor man\'s very long Brownian path:')
    for n in [0,1,10,100,1000,10000,100000,1000000, 10000000]:
        path = [0.0] * (n + 1)
        sigma = 1. / math.sqrt(max(n,1))
        for k in range(1, n + 1):
            path[k] = path[k - 1] + random.choice([-1, 1]) * sigma
        dist = lambda a, b: abs(path[b] - path[a])

        p = 2.25
        pv_start = time.time()
        pv = p_var_backbone(len(path), p, dist)
        pv_time = time.time() - pv_start
        print(f'{n:10d} steps: {p:5.2f}-variation: {pv.value:7.2f}, sequence length: {len(pv.points):5d}, time: {pv_time:7.2f}')

#if __name__ == "__main__":
#    ex_sq()
#    ex_bm()
#    ex_bm_long()


#length, dim = 299, 5
#X = 10.*brownian(length, dim)

## dist = lambda a, b: np.sum(np.abs(X[b,:] - X[a,:]))
#dist = lambda a, b: np.sum((X[b,:] - X[a,:])**2)

#pv_ref = p_var_backbone_ref(path_size=len(X), p=1.2, path_dist=dist)

#M = 100.
#rescaled_path = (M/pv_ref)*X
#rescaled_sig = iisignature.sig(rescaled_path, truncation)
#norm = np.sqrt(np.dot(rescaled_sig, rescaled_sig)+1)
#norm_ker = np.sqrt(sig_kernel(rescaled_path, rescaled_path, n=0))

#norms = []
#norms_ker = []
#flag = True
#for k in np.arange(7, 20, 0.3):
#    rescaled_path = float(1./k)*X
#    rescaled_sig = iisignature.sig(rescaled_path, truncation)
#    norms.append(np.sqrt(np.dot(rescaled_sig, rescaled_sig)+1))
#    norms_ker.append(np.sqrt(sig_kernel(rescaled_path, rescaled_path, n=0)))
#    if np.abs(norms[-1] - norms_ker[-1]) < 1e-2 and flag:
#        print(float(1./k))
#        flag=False


#plt.figure(figsize=(8,6))
#plt.plot(norms, label='naive norm')
#plt.plot(norms_ker, label='kernel norm')
#plt.legend()
#plt.show()