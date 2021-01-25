import os
import multiprocessing
import tempfile
import shutil
from joblib import Parallel, delayed

import numpy as np
import random
import copy
import iisignature
from tools import brownian, brownian_perturbed
from transformers import AddTime 
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def treeEsig(tree, truncation):
    """Function computing the (truncated) Expected Signature of a tree"""
    w, path = tree['value']
    path_dim = path.shape[1]
    forest = tree['forest']
    ES = iisignature.sig(path, truncation)
    if forest:
        ES_forest = sum([treeEsig(t, truncation) for t in forest])
        ES = iisignature.sigcombine(ES, ES_forest, path_dim, truncation)
    return w*ES


def treeMahalanobisDistance(list_of_trees1, list_of_trees2, truncation):
    X = [treeEsig(t, truncation) for t in list_of_trees1]
    V = np.cov(np.array(X).T)
    try:
        Vi = np.linalg.inv(V)
    except:
        Vi = np.linalg.pinv(V)
    Y = [treeEsig(t, truncation) for t in list_of_trees2]
    cd = pairwise_distances(X, Y, 'mahalanobis', VI=Vi, n_jobs=-1)
    return cd


def treeDistance(tree1, tree2, truncation):
    """Function computing the distance between the (truncated) Expected Signatures of two trees"""
    ES1 = treeEsig(tree1, truncation)
    ES2 = treeEsig(tree2, truncation)
    return np.sqrt(np.sum((ES1-ES2)**2))


def processing(j, storing_matrix, list_of_trees1, list_of_trees2, truncation):
    for i in range(j, len(list_of_trees1)):
        storing_matrix[i,j] = treeDistance(list_of_trees1[i], list_of_trees2[j], truncation)

def processing_(j, storing_matrix, list_of_trees1, list_of_trees2, truncation):
    for i in range(len(list_of_trees1)):
        storing_matrix[i,j] = treeDistance(list_of_trees1[i], list_of_trees2[j], truncation)

def treeDistanceMatrix(list_of_trees, truncation):
    """Function computing the pairwise (truncated) ESig distance matrix of elements of one list of trees."""
    
    ## Creat a temporary directory and define the storing matrix
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    storing_matrix = np.memmap(filename, dtype=float, shape=(len(list_of_trees), len(list_of_trees)), mode='w+')
    storing_matrix[:,:] = 0.

    # compute pairwise distances
    Parallel(n_jobs=multiprocessing.cpu_count(), 
             max_nbytes=None, verbose=1)(delayed(processing)(j, storing_matrix, 
                                            list_of_trees, list_of_trees, truncation, 
                                            ) for j in range(len(list_of_trees)))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass

    return storing_matrix

def treePairwiseDistanceMatrix(list_of_trees1, list_of_trees2, truncation):
    """Function computing the pairwise (truncated) ESig distance matrix of two lists of trees."""

    # Creat a temporary directory and define the storing matrix
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    storing_matrix = np.memmap(filename, dtype=float, shape=(len(list_of_trees1), len(list_of_trees2)), mode='w+')
    storing_matrix[:,:] = 0.

    # compute pairwise distances
    Parallel(n_jobs=multiprocessing.cpu_count(), 
             max_nbytes=None, verbose=1)(delayed(processing_)(j, storing_matrix, 
                                            list_of_trees1, list_of_trees2, truncation, 
                                            ) for j in range(len(list_of_trees2)))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass

    return storing_matrix


def generate_brownian_tree(depth, dim, value, min_branches=2, max_branches=5, min_steps=50, max_steps=500, rand=False, perturb=False):
    """Function generating a Brownian tree of the form of a dictionary
       
       return:
           P = {'value': (weight, path), 'forest': [P1, ..., PN]) 
           
       input:
           depth (int): depth of the tree.
           dim (int): dimension of the brownian pathlets.
           value (tuple): root node information (float, array) := (weight, path)
           min_branches, max_branches (both int): min and max amount of branches at every tree-splitting.
           min_steps, max_steps (both int): min and max number of steps in Brownian pathlets.
           rand (bool) if True random weights, otherwise uniform weights.
           perturb (bool) if True perturb Brownian paths.
       """
    
    if depth==1:
        forest = []
    else:
        N = random.randint(min_branches,max_branches)
        steps = random.randint(min_steps,max_steps)
        if rand:
            weights = [random.random() for i in range(N)]
        else:
            weights = [1. for i in range(N)]
        weights = [w/sum(weights) for w in weights]
        if perturb:
            paths = [brownian_perturbed(steps,dim,amplitude=np.random.uniform(-2,2)) for k in range(N)]
        else:
            paths = [brownian(steps,dim) for k in range(N)]
        paths = AddTime().fit_transform(paths)
        paths = [p + value[1][-1,:] for p in paths]
        forest =  [generate_brownian_tree(depth-1, dim, (w,p), min_branches, max_branches, 
                                          min_steps, max_steps, rand, perturb) for w,p in zip(weights, paths)]
        
    return {'value':value, 'forest':forest}


def tree_plot(tree, depth, ax=None, figsize=(14,7)):
    """Function plotting the first dimension of the tree"""
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca(alpha=0.5)
        ax.set_xlabel('time')
        ax.set_title('1-dimensional Brownian tree')
        ax.plot(tree['value'][1][:,0], tree['value'][1][:,1], alpha=0.8)
        ax.plot(tree['value'][1][-1,0], tree['value'][1][-1,1], marker='o', markersize=5, color='blue')
    if depth==1:
        return ax
    else:
        for t in tree['forest']:
            ax.plot(t['value'][1][:,0], t['value'][1][:,1], alpha=0.8)
            if depth>2:
                ax.plot(t['value'][1][-1,0], t['value'][1][-1,1], marker='o', markersize=5, color='blue')
            tree_plot(t,depth-1,ax)


def tree_plot3D(tree, depth, ax=None, figsize=(14,7)):
    """Function plotting the first dimension of the tree"""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('time')
        ax.set_ylabel('BM dim 1')
        ax.set_zlabel('BM dim 1')
        ax.set_title('2-dimensional Brownian tree')
        ax.plot3D(tree['value'][1][:,0], tree['value'][1][:,1], tree['value'][1][:,2], alpha=0.5)
        ax.plot3D([tree['value'][1][-1,0]], [tree['value'][1][-1,1]], [tree['value'][1][-1,2]], marker='o', color='blue')
    if depth==1:
        return ax
    else:
        for t in tree['forest']:
            ax.plot3D(t['value'][1][:,0], t['value'][1][:,1], t['value'][1][:,2], alpha=0.5)
            if depth>2:
                ax.plot3D([t['value'][1][-1,0]], [t['value'][1][-1,1]], [t['value'][1][-1,2]], marker='o', color='blue')
            tree_plot3D(t,depth-1,ax)


def extract_paths_from_tree(tree, list_of_paths=[]):
    tree_ = copy.deepcopy(tree)
    w, path = tree_['value']
    forest = tree_['forest']
    if not forest:
        list_of_paths.append(path)
    else:
        for t in forest:
            w, p = t['value']
            p = np.concatenate([path[:-1,:], p], axis=0)
            t['value'] = copy.deepcopy((w, p))
            extract_paths_from_tree(t, list_of_paths)
    return list_of_paths