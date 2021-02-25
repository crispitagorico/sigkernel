import numpy as np
import copy
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array

def normalize(sigs, width, depth):
    new_sigs = []
    for sig in tqdm(sigs):
        new_sig = np.zeros_like(sig)
        for k in range(depth):
            dim = width*(width**(k)-1)
            new_sig[dim:dim + width**(k+1)] = math.factorial(k+1)*sig[dim:dim + width**(k+1)]
        new_sigs.append(new_sig)
    return new_sigs

class AddTime(BaseEstimator, TransformerMixin):
    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

class Reversion(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [as_float_array(x[::-1]) for x in X]


class LeadLag(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        lag = []
        lead = []

        for val_lag, val_lead in zip(X[:-1], X[1:]):
            lag.append(val_lag)
            lead.append(val_lag)

            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(X[-1])
        lead.append(X[-1])

        return np.c_[lag, lead]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

class Dyadic(BaseEstimator, TransformerMixin):
    def __init__(self, depth):
        self.depth = depth

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        n_splits = 2**self.depth

        intervals = np.array_split(X, n_splits)

        for i in range(1, len(intervals)):
            intervals[i] = np.r_[[intervals[i - 1][-1]], intervals[i]]

        return [as_float_array(interval) for interval in intervals]

    def transform(self, X, y=None):
        return [self.transform_instance(x) for x in X]

class PenOff(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):

        # Add pen-off
        X_transformed = np.c_[X, np.ones(len(X))]
        last = np.array(copy.deepcopy(X_transformed[-1]))
        last[-1] = 0.

        X_transformed = np.r_[X_transformed, [last]]

        # Add home
        X_transformed = np.r_[np.zeros(X_transformed.shape[1]).reshape(1, -1),
                              X_transformed]

        return X_transformed

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]

class Stroke_Augment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        stroke = 0.
        output = []
        for c in X:
            output += [r + [stroke] for r in c]
            stroke += 1.
        return np.array(output)

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]

class Ink_Augment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        output = []
        ink = 0.
        for c in X:
            for d in c:
                output += d + [ink]
                if d != c[-1]:
                    ink += 1.
        return np.array(output).reshape(-1, 3)

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]

class Pen_Augment(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        output = []
        for c in X:
            output += [c[0] + [1.]]
            output += [r + [0.] for r in c]
            output += [c[-1] + [1.]]
        return np.array(output[1:-1])

    def transform(self, X, Y=None):
        return [self.transform_instance(x) for x in X]
