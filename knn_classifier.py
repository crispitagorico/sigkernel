import os
import multiprocessing
import tempfile
import shutil
from joblib import Parallel, delayed

import numpy as np
from scipy.stats import mode

from tslearn.metrics import dtw, soft_dtw
from sigKer_fast import sig_distance
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

def reparametrization(length, sub_rate, replace):
    a = np.random.choice(range(length), size=int(sub_rate*length), replace=replace)
    a.sort()
    return [0] + a.tolist() + [length-1]

def processing(j, dm, x_train, x_test, length, metric, sub_rate, replace, gamma, n):
    reparam = reparametrization(length, sub_rate, replace=replace)
    for i in range(len(x_train)):
        if metric=='dtw':
            dm[i,j] = dtw(x_train[i][reparam,:], x_test[j][reparam,:])
        elif metric=='soft_dtw':
            dm[i,j] = soft_dtw(x_train[i][reparam,:], x_test[j][reparam,:], gamma)
        elif metric == 'sig':
            dm[i,j] = sig_distance(x_train[i][reparam,:], x_test[j][reparam,:], n)
            pass
        
def knn_classifier(x_train, x_test, y_train, y_test, sub_rate, length, 
                   metric='dtw', replace=False, n_neighbours=1, gamma=0.1, n=0):
    
    ## Creat a temporary directory and define the array path
    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'joblib.mmap')
    dm = np.memmap(filename, dtype=float, shape=(len(x_train), len(x_test)), mode='w+')
    
    # compute distances
    Parallel(n_jobs=multiprocessing.cpu_count(), 
             max_nbytes=None, 
             verbose=0)(delayed(processing)(j, 
                                            dm, 
                                            x_train, 
                                            x_test, 
                                            length, 
                                            metric,
                                            sub_rate, 
                                            replace,
                                            gamma, 
                                            n,
                                            ) for j in range(len(x_test)))

    #Delete the temporary directory and contents
    try:
        shutil.rmtree(temp_folder)
    except OSError:
        pass
    
    knn_idx = dm.T.argsort()[:, :n_neighbours]
    knn_labels = y_train[knn_idx]

    mode_data = mode(knn_labels, axis=1)
    mode_label = mode_data[0]
    mode_proba = mode_data[1]/n_neighbours

    label = mode_label.ravel()
    proba = mode_proba.ravel()
    
    conf_mat = confusion_matrix(label, y_test)
    conf_mat = conf_mat/conf_mat.sum(0)
    
    acc_score = accuracy_score(label, y_test)

    return label, proba, acc_score, conf_mat