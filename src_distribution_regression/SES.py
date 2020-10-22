import numpy as np
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn_transformers import AddTime, LeadLag, pathwiseExpectedSignatureTransform, SignatureTransform
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline


def model(X, y, depths1=[2], depth2=2, ll=None, at=False, NUM_TRIALS=5, cv=3, grid={}):
    """Performs a Lasso-based distribution regression on ensembles (of possibly unequal cardinality)
       of univariate or multivariate time-series (of possibly unequal lengths)

       Input: depths1 (list of ints): truncation of the signature 1 (is cross-validated)
              depth2 (int): truncation of the second signature

              X (list): list of lists such that

                        - len(X) = n_samples

                        - for any i, X[i] is a list of arrays of shape (length, dim)

                        - for any j, X[i][j] is an array of shape (length, dim)

              y (np.array): array of shape (n_samples,)

              ll (list of ints): dimensions to lag
              at (bool): if True pre-process the input path with add-time

              NUM_TRIALS, cv (int): parameters for nested cross-validation

              grid (dict): a dictionary to specify the hyperparameter grid for the gridsearch. Unspecified entries will be set by default

       Output: mean MSE (and std) (both scalars) of regression performance on a cv-folds cross-validation (NUM_TRIALS times)

    """
    
    if X[0][0].shape[1] == 1:
        assert ll is not None or at == True, "must add one dimension to the time-series, via ll=[0] or at=True"
        
    # possibly augment the state space of the time series
    if ll is not None:
        X = LeadLag(ll).fit_transform(X)
    if at:
        X = AddTime().fit_transform(X)
  

    # parameters for grid search
    parameters = {'lin_reg__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
                  'lin_reg__fit_intercept': [False, True],
                  'lin_reg__normalize': [True, False],
                  }

    # check if the user has not given an irrelevant entry
    assert len(list(set(parameters.keys()) & set(grid.keys()))) == len(
        list(grid.keys())), "keys should be in " + ' '.join([str(e) for e in parameters.keys()])

    # merge the user grid with the default one
    parameters.update(grid)

    pipe = Pipeline([('lin_reg', Lasso(max_iter=1000))])

    scores = np.zeros(NUM_TRIALS)
    results = {}
    # Loop for each trial
    for i in tqdm(range(NUM_TRIALS)):

        best_scores_train = np.zeros(len(depths1))

        # will only retain the MSE (mean + std) corresponding to the model achieving the best score on the train set
        # i.e. the test set is not used to decide the truncation level.
        MSE_test = np.zeros(len(depths1))
        results_tmp = {}
        for n, depth in enumerate(depths1):
            pwES = pathwiseExpectedSignatureTransform(order=depth).fit_transform(X)
            SpwES = SignatureTransform(order=depth2).fit_transform(pwES)

            X_train, X_test, y_train, y_test = train_test_split(np.array(SpwES), np.array(y), test_size=0.2,
                                                                random_state=i)

            # parameter search
            model = GridSearchCV(pipe, parameters, verbose=0, n_jobs=-1, scoring='neg_mean_squared_error', cv=cv,
                                 error_score=np.nan)

            model.fit(X_train, y_train)
            best_scores_train[n] = -model.best_score_

            y_pred = model.predict(X_test)
            results_tmp[n]={'pred':y_pred,'true':y_test}
            MSE_test[n] = mean_squared_error(y_pred, y_test)

        # pick the model with the best performances on the train set
        best_score = 100000
        index = None
        for n, depth in enumerate(depths1):
            if (best_scores_train[n] < best_score):
                best_score = best_scores_train[n]
                index = n

        scores[i] = MSE_test[index]
        results[i] = results_tmp[index]
        print('best truncation level (cv on train set): ', depths1[index])
    return scores.mean(), scores.std(), results