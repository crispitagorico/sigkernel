<h1 align='center'>The Signature Kernel as the solution of a Goursat PDE
    [<a href="https://arxiv.org/abs/2006.14794">arXiv</a>] </h1>

## Overview

The Signature Kernel is a new learning tool designed to handle irregularly sampled, multidimensional time series. Here we show that the Signature Kernel solves an hyperbolic PDE and recognize the link with a well known class of differential equations known in the literature as Goursat problems. This Goursat PDE only depends on the increments of the input sequences, does not require the explicit computation of signatures and can be solved using state-of-the-art hyperbolic PDE numerical solvers; it is effectively a kernel trick for the Signature Kernel. 

 h<sub>&theta;</sub>(x) = &theta;<sub>o</sub> x + &theta;<sub>1</sub>x

-----

## Paper
This repository contains CPU and GPU implementations for the Signature Kernel trick, i.e. finite difference schemes to solve the Goursat PDE. It also contains all the code for reproducing the experiments from the <a href="https://arxiv.org/abs/2006.14794">The Signature Kernel as the solution of a Goursat PDE</a> paper.

-----

## Code

### Setup the Environment
To setup the environment, install the requirements with

+ `pip install -r requirements.txt`

Next install pytorch by running only of the following options

#### CPU only

+ `pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

#### CUDA 10.2

+ `pip install torch==1.6.0 torchvision==0.7.0`

#### CUDA 10.1

+ `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

#### CUDA 9.2

+ `pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html`

-----

### Build CPU PDE solver (Cython)
Navigate into `src/` and run

+ `python setup.py build_ext --inplace`

### Run Experiments
All the experiments can be found in `notebooks/`. 

To reproduce the error distribution plots and time performance analysis execute all cells in `notebooks/performance_analysis.ipynb`

To reproduce the time series classification results execute all cells in `notebooks/time_series_classification.ipynb`. The results will also be saved to `results/`.

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

