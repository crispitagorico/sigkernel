import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import iisignature
import torch 
import math
import pickle
from time import sleep

from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import GridSearchCV

from tslearn.svm import TimeSeriesSVC
from sklearn.svm import SVC

import sigkernel

_datasets = [
            'ArticularyWordRecognition', 
            'BasicMotions', 
            'Cricket',
            'ERing',
            'Libras', 
            'NATOPS', 
            'RacketSports',     
            'FingerMovements',
            'Heartbeat',
            'SelfRegulationSCP1', 
            'UWaveGestureLibrary'
            ]

_kernels =  [
#             'linear',
#             'rbf',
#             'gak',
            'truncated signature',
#             'signature pde'
            ]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', help='True to retrain, False otherwise.', action='store_true')
    parser.add_argument('-te', '--test', help='True to repeat test, False otherwise.', action='store_true')
    parser.add_argument('-p', '--print', help='True to print the latest test results, False otherwise.', action='store_true')
    args = parser.parse_args()
    
    #==================================================================================================
    # Training phase
    #==================================================================================================
    if args.train:

        # store best models in training phase 
        try:
            with open('../results/trained_models.pkl', 'rb') as file:
                trained_models = pickle.load(file)    
        except:
            trained_models = {}    

        # define grid-search hyperparameters for SVC (common to all kernels)
        svc_parameters = {'C': np.logspace(0, 4, 5), 'gamma': list(np.logspace(-4, 4, 9)) + ['auto']}
        _sigmas = [1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1., 2., 5., 10.]
        _scales = [5e-2, 1e-1, 5e-1, 1e0]
            
        # start grid-search
        datasets = tqdm(_datasets, position=0, leave=True)
        for name in datasets:
            
            # record best scores in training phase
            best_scores_train = {k : 0. for k in _kernels}

            # lead-lag only if number of channels is <= 5
            x_train, _, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)    
            if x_train.shape[1] <= 200 and x_train.shape[2] <= 8: 
                transforms = tqdm([(True,True), (False,True), (True,False), (False,False)], position=1, leave=False)
            else: # do not try lead-lag as dimension is already high
                transforms = tqdm([(True,False), (False,False)], position=1, leave=False)
                
            # grid-search for path-transforms (add-time, lead-lag)
            for (at,ll) in transforms:
                transforms.set_description(f"add-time: {at}, lead-lag: {ll}")

                # load train data
                x_train, y_train, _, _ = UCR_UEA_datasets(use_cache=True).load_dataset(name)
                x_train /= x_train.max()

                # encode outputs as labels
                y_train = LabelEncoder().fit_transform(y_train)

                # path-transform
                x_train = sigkernel.transform(x_train, at=at, ll=ll, scale=.1)

                # subsample every time steps if certain length is exceeded
                subsample = max(int(np.floor(x_train.shape[1]/149)),1)
                x_train = x_train[:,::subsample,:]
                datasets.set_description(f"dataset: {name} --- shape: {x_train.shape}")

                #==================================================================================
                # Linear, RBF and GAK kernels
                #==================================================================================
                # define standard kernels
                std_kernels = tqdm(['linear', 'rbf', 'gak'], position=2, leave=False)
                for ker in std_kernels:
                    std_kernels.set_description(f"standard kernel: {ker}")

                    # SVC tslearn estimator
                    svc = TimeSeriesSVC(kernel=ker, decision_function_shape='ovo')
                    svc_model = GridSearchCV(estimator=svc, param_grid=svc_parameters, cv=5, n_jobs=-1)
                    svc_model.fit(x_train, y_train)
                    
                    # store results
                    if svc_model.best_score_ > best_scores_train[ker]:
                        best_scores_train[ker] = svc_model.best_score_
                        trained_models[(name, ker)] = (subsample, at, ll, svc_model)

                    sleep(0.5)

                #==================================================================================
                # Truncated signature kernels
                #==================================================================================
                # set max signature truncation
                dim  = x_train.shape[-1]
                if dim <= 4:
                    max_depth = 6
                elif dim <= 6:
                    max_depth = 5
                elif dim <= 8:
                    max_depth = 4
                else:
                    max_depth = 3

                # grid search on truncation levels
                depths = tqdm(range(2,max_depth+1), position=2, leave=False)
                for depth in depths:
                    depths.set_description(f"truncated signature depth: {depth}")

                    scales = tqdm(_scales, position=3, leave=False)
                    for scale in scales:
                        scales.set_description(f"truncated signature scale: {scale}")
                    
                        # truncated signatures
                        sig_train = iisignature.sig(scale*x_train, depth)
                        
                        for ker_ in ['linear', 'rbf']:
                            for normalize in [True, False]:
                            
                                # normalization
                                if normalize:
                                    sig_train = sigkernel.normalize(sig_train, x_train.shape[-1], depth)
                        
                                # SVC tslearn estimator
                                svc = SVC(kernel=ker_, decision_function_shape='ovo')
                                svc_model = GridSearchCV(estimator=svc, param_grid=svc_parameters, cv=5, n_jobs=-1)
                                svc_model.fit(sig_train, y_train)
                        
                                # store results
                                if svc_model.best_score_ > best_scores_train['truncated signature']:
                                    best_scores_train['truncated signature'] = svc_model.best_score_
                                    trained_models[(name, 'truncated signature')] = (subsample, at, ll, depth, scale, 
                                                                                     ker_, normalize, svc_model)

                        sleep(0.5)

                #==================================================================================
                # Signature PDE kernel
                #==================================================================================
                # move to cuda (if available and memory doesn't exceed a certain threshold)
                if x_train.shape[0] <= 150 and x_train.shape[1] <=150 and x_train.shape[2] <= 8:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    dtype = torch.float32
                else: # otherwise do computations in cython
                    device = 'cpu'
                    dtype = torch.float64
                
                # numpy -> torch
                x_train = torch.tensor(x_train, dtype=dtype, device=device)
                
                # grid search over sigmas
                sigmas = tqdm(_sigmas, position=2, leave=False)
                for sigma in sigmas:
                    sigmas.set_description(f"signature PDE sigma: {sigma}")

                    # define static kernel
                    static_kernel = sigkernel.RBFKernel(sigma=sigma)

                    # initialize corresponding signature PDE kernel
                    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)

                    # compute Gram matrix on train data
                    G_train = signature_kernel.compute_Gram(x_train, x_train, sym=True).cpu().numpy()

                    # SVC sklearn estimator
                    svc = SVC(kernel='precomputed', decision_function_shape='ovo')
                    svc_model = GridSearchCV(estimator=svc, param_grid=svc_parameters, cv=5, n_jobs=-1)
                    svc_model.fit(G_train, y_train)
                    
                    # empty memory
                    del G_train
                    torch.cuda.empty_cache()
        
                    # store results
                    if svc_model.best_score_ > best_scores_train['signature pde']:
                        best_scores_train['signature pde'] = svc_model.best_score_
                        trained_models[(name, 'signature pde')] = (subsample, at, ll, sigma, svc_model)

                    sleep(0.5)

            # save trained models
            with open('../results/trained_models.pkl', 'wb') as file:
                pickle.dump(trained_models, file)


    #==================================================================================================
    # Testing phase
    #==================================================================================================
    if args.test:

        # load trained models
        try:
            with open('../results/trained_models.pkl', 'rb') as file:
                trained_models = pickle.load(file)
        except:
            print('Models need to be trained first')

        # load final results from last run 
        try:
            with open('../results/final_results.pkl', 'rb') as file:
                final_results = pickle.load(file)
        except:
            final_results = {}

        for name in _datasets:            
            for ker in _kernels:
            
                # load test data
                x_train, y_train, x_test, y_test = UCR_UEA_datasets(use_cache=True).load_dataset(name)
                x_train /= x_train.max()
                x_test /= x_test.max()
                
                # encode outputs as labels
                y_test = LabelEncoder().fit_transform(y_test)
                
                #==================================================================================
                # Linear, RBF and GAK kernels
                #==================================================================================
                if ker in ['linear', 'rbf', 'gak']:
                    
                    # extract information from training phase
                    subsample, at, ll, estimator = trained_models[(name,ker)]
                    
                    # path-transform and subsampling
                    x_test = sigkernel.transform(x_test, at=at, ll=ll, scale=.1)[:,::subsample,:]
                    
                    # record scores
                    train_score = estimator.best_score_
                    test_score = estimator.score(x_test, y_test)
                    final_results[(name,ker)] = {f'training accuracy: {train_score} %', f'testing accuracy: {test_score} %'}
                    
                #==================================================================================
                # Truncated signature kernel
                #==================================================================================
                elif ker == 'truncated signature':
                    
                    # extract information from training phase
                    subsample, at, ll, depth, scale, ker_, normalize, estimator = trained_models[(name,ker)]
                    
                    # path-transform and subsampling
                    x_test = sigkernel.transform(x_test, at=at, ll=ll, scale=scale*.1)[:,::subsample,:]
                    
                    # truncated signatures
                    sig_test = iisignature.sig(x_test, depth)
                    
                    # normalization
                    if normalize:
                        sig_test = sigkernel.normalize(sig_test, x_test.shape[-1], depth)
                    
                    # record scores
                    train_score = estimator.best_score_
                    test_score = estimator.score(sig_test, y_test)
                    final_results[(name,ker)] = {f'training accuracy: {train_score} %', f'testing accuracy: {test_score} %'}
                    
                #==================================================================================
                # Signature PDE kernel
                #==================================================================================
                else:
                    assert ker == 'signature pde'
                    
                    # extract information from training phase
                    subsample, at, ll, sigma, estimator = trained_models[(name,ker)]
                    
                    # path-transform and subsampling
                    x_train = sigkernel.transform(x_train, at=at, ll=ll, scale=.1)[:,::subsample,:]
                    x_test = sigkernel.transform(x_test, at=at, ll=ll, scale=.1)[:,::subsample,:]
                    
                    # move to cuda (if available and memory doesn't exceed a certain threshold)
                    if x_test.shape[0] <= 150 and x_test.shape[1] <=150 and x_test.shape[2] <= 10:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        dtype = torch.float32
                    else: # otherwise do computations in cython
                        device = 'cpu'
                        dtype = torch.float64
                        
                    # numpy -> torch 
                    x_train = torch.tensor(x_train, dtype=dtype, device=device)
                    x_test = torch.tensor(x_test, dtype=dtype, device=device)
                    
                    # define static kernel
                    static_kernel = sigkernel.RBFKernel(sigma=sigma)
                    
                    # initialize corresponding signature PDE kernel
                    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)
                        
                    # compute Gram matrix on test data
                    G_test = signature_kernel.compute_Gram(x_test, x_train, sym=False).cpu().numpy()
                    
                    # record scores
                    train_score = estimator.best_score_
                    test_score = estimator.score(G_test, y_test)
                    final_results[(name,ker)] = {f'training accuracy: {train_score} %', f'testing accuracy: {test_score} %'}

                    # empty memory
                    del G_test
                    torch.cuda.empty_cache()

                sleep(0.5)
                
                if ker == 'truncated signature':
                    print(name, ker, f'best truncation: {depth}', f'best kernel: {ker_}', final_results[name,ker])
                else:
                    print(name, ker, final_results[name,ker])
                
            print('\n')
        
        # save results
        with open('../results/final_results.pkl', 'wb') as file:
            pickle.dump(final_results, file)
    
    #==================================================================================================
    # Print latest test results
    #==================================================================================================
    if args.print:
        with open('../results/final_results.pkl', 'rb') as file:
            final = pickle.load(file)
        print(final)
        



