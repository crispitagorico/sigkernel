import sys
sys.path.append('../src')

import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn_transformers import AddTime, LeadLag

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sigKer_fast import sig_mmd_distance as sig_kernels_mmd

from joblib import Parallel, delayed

def model(X, y, scales=[1], ll=None, at=False, mode='krr', NUM_TRIALS=5, cv=3, grid={}):
    """Performs a kernel based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)

       Input:
              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)

              scales (list of floats): time-series scaling parameter to cross-validate
              ll (list of ints): dimensions to lag (set to None by default)
              at (bool): if True pre-process the input path with add-time

              mode (str): "krr" -> Kernel Ridge Regression, 'svr' -> Support Vector Regresion

              NUM_TRIALS, cv : parameters for cross-validation

              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default

       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times) as well results (a dictionary containing the predicted labels and true labels)
    """

    assert mode in ['svr', 'krr'], "mode must be either 'svr' or 'krr' "

    if X[0][0].shape[1] == 1:
        assert ll is not None or at == True, "must add one dimension to the time-series, via ll=[0] or at=True"
        
    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
    
    if mode == 'krr':

        # default grid
        parameters = {'clf__kernel': ['precomputed'],
                      'clf__gamma': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2,1e3],
                      'clf__alpha': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)

        clf = KernelRidge

    else:

        # default grid
        parameters = [{'clf__kernel': ['precomputed'],
                       'clf__gamma': [gamma(1e-3),gamma(1e-2), gamma(1e-1), gamma(1), gamma(1e1), gamma(1e2),gamma(1e3)], 
                       'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
                       }]

        # check if the user has not given an irrelevant entry
        assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
            list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

        # merge the user grid with the default one
        parameters.update(grid)
        clf = SVR

    list_kernels = []

    # Precompute the Gram matrices for the different scaling parameters, to avoid recomputing them for each grid search step
    for scale in tqdm(scales):
        K_full = np.zeros((len(X), len(X)),dtype=np.float64)
        indices = np.triu_indices(len(X),k=0,m=len(X))
        K_full[indices] = Parallel(n_jobs=-1,verbose=3)(
            delayed(ExpectedKernel)(X[i],X[j],sym=False,scale=scale,n=0)
            for i in range(len(X))
            for j in range(i,len(X))
        ) 
        indices = np.tril_indices(len(X), k=-1, m=len(X))
        K_full[indices] = K_full.T[indices]
        
        diag = np.diag(K_full)
        mmd = -2. * K_full + np.tile(diag,(K_full.shape[0],1)) + np.tile(diag[:,None],(1,K_full.shape[0]))
        list_kernels.append(mmd)

    
    scores = np.zeros(NUM_TRIALS)
    results = {}
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(scales))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score (on the train set)
        # i.e. the test set is not used to decide the hyperparameters.
   
        MSE_test = np.zeros(len(scales))
        results_tmp = {}
        for n, scale in enumerate(scales):
            
            ind_train, ind_test, y_train, y_test = train_test_split(np.arange(len(y)), np.array(y), test_size=0.2,
                                                                random_state=i)

            # building the estimator
            pipe = Pipeline([('rbf_mmd', RBF_Sig_MMD_Kernel(K_full=list_kernels[n])),
                    ('clf', clf())
                    ])
            # parameter search
            model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(ind_train, y_train)
            best_scores_train[n] = -model.best_score_

            y_pred = model.predict(ind_test)
        
            results_tmp[n]={'pred':y_pred,'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)

        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n, scale in enumerate(scales):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
        print('best scaling parameter (cv on the train set): ', scales[index])
    
    return scores.mean(), scores.std(), results


class RBF_Sig_MMD_Kernel(BaseEstimator, TransformerMixin):
    def __init__(self, K_full=None,gamma=1.0):
        super(RBF_Sig_MMD_Kernel, self).__init__()
        self.gamma = gamma
        self.K_full = K_full


    def transform(self, X):
        alpha = 1. / (2 * self.gamma ** 2)
        K = self.K_full[X][:,self.ind_train].copy()
        return np.exp(-alpha*K) 

    def fit(self, X, y=None, **fit_params):
        self.ind_train = X
        return self

def ExpectedKernel(X_i,X_j,sym,scale,n=0):
    tree_i = np.array([scale*branch for branch in X_i],dtype=np.float64)
    tree_j = np.array([scale*branch for branch in X_j],dtype=np.float64)
    K_ij = sig_kernels_mmd(tree_i,tree_j,sym=False,n=n) # increasing n corresponds to increasing the number of steps
    # taken by the PDE solver (forward finite-difference scheme)
    return np.mean(K_ij[:,:,-1,-1])  
