<h1 align='center'>The Signature Kernel as the solution of a Goursat PDE
    [<a href="https://arxiv.org/abs/2006.14794">arXiv</a>] </h1>

## Overview

The Signature Kernel is a new learning tool designed to handle irregularly sampled, multidimensional time series. Here we show that the Signature Kernel solves an hyperbolic PDE and recognize the link with a well known class of differential equations known in the literature as Goursat problems. This Goursat PDE only depends on the increments of the input sequences, does not require the explicit computation of signatures and can be solved using state-of-the-art hyperbolic PDE numerical solvers; it is effectively a kernel trick for the Signature Kernel. 

-----

## Paper
This repository contains CPU and GPU implementations for the Signature Kernel trick, i.e. finite difference schemes to solve the Goursat PDE. It also contains all the code for reproducing the experiments from the <a href="https://arxiv.org/abs/2006.14794">The Signature Kernel as the solution of a Goursat PDE</a> paper.

-----

## Code

### Setup the Environment
To setup the environment, install the requirements with

+ `pip install -r requirements.txt`

### Build CPU PDE solver (Cython)
Navigate into `src/` and run

+ `python setup.py build_ext --inplace`

### Run the Experiments
All the experiments can be found in `notebooks/`. 

To reproduce the time series classification results execute all cells in `notebooks/time_series_classification.ipynb`. The results will also be save to `results/`.

To reproduce the time series prediction results and plots exectue all cells in `notebooks/prediction_bitcoin_prices.ipynb`

## Citation

<!-- 
```bibtex
@article{morrill2020logode,
    author={Morrill, James and Kidger, Patrick and Salvi, Cristopher and Foster, James and Lyons, Terry},
    title={{Neural CDEs for Long Time-Series via the Log-ODE Method}},
    year={2020},
    journal={arXiv:2009.08295}
}
```
-->

