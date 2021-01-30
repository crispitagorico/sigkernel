<h1 align='center'>The Signature Kernel is the solution of a Goursat PDE
    [<a href="https://arxiv.org/abs/2006.14794">arXiv</a>] </h1>

## Overview

The Signature Kernel is a new learning tool designed to handle irregularly sampled, multidimensional time series. In this <a href="https://arxiv.org/abs/2006.14794">paper</a> we show that the Signature Kernel solves an hyperbolic PDE and recognize the link with a well known class of differential equations known in the literature as Goursat problems. This Goursat PDE only depends on the increments of the input sequences, does not require the explicit computation of signatures and can be solved using state-of-the-art hyperbolic PDE numerical solvers; it is effectively a kernel trick for the Signature Kernel. 

-----

## Paper
This repository contains CPU and GPU implementations of various finite difference schemes to solve the Signature Kernel PDE. It also contains all the code for reproducing the experiments from the <a href="https://arxiv.org/abs/2006.14794">The Signature Kernel as the solution of a Goursat PDE</a> paper.

-----

## Code

### Setup the Environment
To setup the environment, install the requirements with

+ `pip install -r requirements.txt`

Next install pytorch 1.6.0 by running one of the following options:

#### CPU only

+ `pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`

#### CUDA 10.2

+ `pip install torch==1.6.0 torchvision==0.7.0`

To install torch 1.6.0 for other versions of CUDA see <a href="https://pytorch.org/get-started/previous-versions/"> torch install website </a>.

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

```bibtex
@article{cass2020computing,
  title={Computing the full signature kernel as the solution of a Goursat problem},
  author={Cass, Thomas and Lyons, Terry and Salvi, Cristopher and Yang, Weixin},
  journal={arXiv preprint arXiv:2006.14794},
  year={2020}
}
```

<!-- 
-->

