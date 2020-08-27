
import sys
import numpy as np
from esig import tosig as ts ## from esig import tosig as ts
from esig import tests as tests
from numpy.linalg import norm
import time
import esig

print(esig.get_version())
tests.run_tests()

test_outcome = 0

dimension = 240 ## big test set this to 2400 small test 240 
no_points = 10000 ## big test set this to 1000000 small test 10000

print(f"preparing data ({no_points},{dimension})")
tic = time.perf_counter()
data = np.random.rand(no_points,dimension)
toc = time.perf_counter()
print(f"prepared data in {toc - tic:0.4f} seconds\n")

## test 1
tic = time.perf_counter()
selected_points, new_weights = ts.recombine(data) ## degree = 1
toc = time.perf_counter()
print(f"Recombined {no_points} points in dimension {dimension} to {dimension+1} points in {toc - tic:0.4f} seconds")

## check mean preserved
old_average=np.sum(data, 0)
new_average=new_weights.dot(np.take(data, selected_points, 0))
normalised_error = norm(old_average - new_average)/(norm(old_average) + norm(new_average))
print("normalised difference in the integrals = ",normalised_error)

## report
if ((len(selected_points) > dimension + 1) or (normalised_error > 1e-11)):
    test_outcome = test_outcome - 1
else:
    print("test passed")

## test2 
### the points are not spanning the full space and so the minimal set should have cardinality less than or equal rank + 1
matrix = np.random.rand(dimension,dimension + 20)
new_data = data.dot(matrix)
tic = time.perf_counter()
selected_points, new_weights = ts.recombine(new_data) ## degree = 1
toc = time.perf_counter()
print(f"\nRecombined {no_points} points in dimension {dimension + 20} to {dimension+1} points in {toc - tic:0.4f} seconds")

## check mean preserved
old_average=np.sum(data, 0)
new_average=new_weights.dot(np.take(data, selected_points, 0))
normalised_error = norm(old_average - new_average)/(norm(old_average) + norm(new_average))
print("normalised difference in the integrals = ",normalised_error)
print("no of points left = ", len(selected_points))

## report
if ((len(selected_points) > dimension + 1) or (normalised_error > 1e-12)):
    test_outcome = test_outcome - 1
else:
    print("test passed")

#test3
## test the degree > 1 case - match second moments
dimension = 10
no_points = 1000
data = np.random.rand(no_points,dimension)

tic = time.perf_counter()
selected_points, new_weights = ts.recombine(data, degree = 2)
toc = time.perf_counter()
print(f"\nRecombined {no_points} points in dimension {dimension} to {len(selected_points)} points in {toc - tic:0.4f} seconds while preserving first and second moments")

old_average=np.sum(data, 0)
new_average=new_weights.dot(np.take(data, selected_points, 0))
normalised_error_in_mean = norm(old_average - new_average)/(norm(old_average) + norm(new_average))

new_cov = np.cov(np.take(data, selected_points, 0),rowvar=False, bias=True, aweights=new_weights)
old_cov = np.cov(data,rowvar=False, bias=True,aweights=np.full(1000,1.))
normalised_error_in_cov = norm(old_cov - new_cov)/(norm(old_cov) + norm(new_cov))

print (f"normalised_error_in_mean {normalised_error_in_mean}\nnormalised_error_in_cov  {normalised_error_in_cov}")

if ((normalised_error_in_mean > 1e-13) or (normalised_error_in_cov > 1e-13)):
    test_outcome = test_outcome - 1
else:
    print("test passed")

print ("\n")

import platform, subprocess

print("="*40, "System Information", "="*40)
uname = platform.uname()
print(f"System: {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

sys.exit(test_outcome)
