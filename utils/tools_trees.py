import numpy as np
import random
import iisignature
from tools import brownian
from transformers import AddTime 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def treeEsig(tree, truncation, path_dim):
    """Function computing the truncated Expected Signature of a tree"""
    w, path = tree['value']
    forest = tree['forest']
    ES = w*iisignature.sig(path, truncation)
    if not forest:
        return ES
    else:
        ES_forest = sum([treeEsig(t, truncation, path_dim) for t in forest])
        ES = iisignature.sigcombine(ES, ES_forest, path_dim, truncation)
        return ES

def generate_brownian_tree(depth, dim, value, min_branches=2, max_branches=3, min_steps=10, max_steps=100):
    """Function generating a Brownian tree of the form of a dictionary
       
       return:
           P = {'value': (weight, path), 'forest': [P1, ..., PN]) 
           
       input:
           depth (int): depth of the tree, 
           dim (int): dimension of the brownian pathlets
           value (tuple): root node information (float, array) := (weight, path)
           min_branches, max_branches (both int): min and max amount of branches at every tree-splitting
           min_steps, max_steps (both int): min and max number of steps in Brownian pathlets
       """
    
    if depth==1:
        return {'value': value, 
                'forest': []}
    else:
        N = random.randint(min_branches,max_branches)
        steps = random.randint(min_steps,max_steps)
        weights = [random.random() for i in range(N)]
        weights = [w/sum(weights) for w in weights]
        paths = [brownian(steps,dim) for k in range(N)]
        paths = AddTime().fit_transform(paths)
        return {'value': value, 
                'forest': [generate_brownian_tree(depth-1,dim,(w, p+value[1][-1,:])) for w,p in zip(weights, paths)]}

def tree_plot(tree, depth, ax=None):
    """Function plotting the first dimension of the tree"""
    if ax is None:
        plt.figure(figsize=(16,10))
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

def tree_plot3D(tree, depth, ax=None):
    """Function plotting the first dimension of the tree"""
    if ax is None:
        fig = plt.figure(figsize=(16,10))
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

