import numpy as np
import copy
import random
import doctest
import iisignature
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import imp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array


class AddTime(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to add time as an extra dimension of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, init_time=0., total_time=1.):
        self.init_time = init_time
        self.total_time = total_time

    def fit(self, X, y=None):
        return self

    def transform_instance(self, X):
        t = np.linspace(self.init_time, self.init_time + 1, len(X))
        return np.c_[t, X]

    def transform(self, X, y=None):
        return [[self.transform_instance(x) for x in bag] for bag in X]


class LeadLag(BaseEstimator, TransformerMixin):
    # sklearn-type estimator to compute the Lead-Lag transform of a D-dimensional path.
    # Note that the input must be a list of arrays (i.e. a list of D-dimensional paths)

    def __init__(self, dimensions_to_lag):
        if not isinstance(dimensions_to_lag, list):
            raise NameError('dimensions_to_lag must be a list')
        self.dimensions_to_lag = dimensions_to_lag

    def fit(self, X, y=None):
        return self

    def transform_instance_1D(self, x):

        lag = []
        lead = []

        for val_lag, val_lead in zip(x[:-1], x[1:]):
            lag.append(val_lag)
            lead.append(val_lag)
            lag.append(val_lag)
            lead.append(val_lead)

        lag.append(x[-1])
        lead.append(x[-1])

        return lead, lag

    def transform_instance_multiD(self, X):
        if not all(i < X.shape[1] and isinstance(i, int) for i in self.dimensions_to_lag):
            error_message = 'the input list "dimensions_to_lag" must contain integers which must be' \
                            ' < than the number of dimensions of the original feature space'
            raise NameError(error_message)

        lead_components = []
        lag_components = []

        for dim in range(X.shape[1]):
            lead, lag = self.transform_instance_1D(X[:, dim])
            lead_components.append(lead)
            if dim in self.dimensions_to_lag:
                lag_components.append(lag)

        return np.c_[lead_components + lag_components].T

    def transform(self, X, y=None):
        return [[self.transform_instance_multiD(x) for x in bag] for bag in X]


class ExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get the lengths of all time series (across items across bags)
        lengths = [item.shape[0] for bag in X for item in bag]
        if len(list(set(lengths))) == 1:
            # if all time series have the same length, the signatures can be computed in batch
            X = [iisignature.sig(bag, self.order) for bag in X]
        else:
            X = [np.array([iisignature.sig(item, self.order) for item in bag]) for bag in X]
        return [x.mean(0) for x in X]


class pathwiseExpectedSignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pwES = []
        for bag in X:
            # get the lengths of all time series in the bag
            lengths = [item.shape[0] for item in bag]
            if len(list(set(lengths))) == 1:
                # if all time series have the same length, the (pathwise) signatures can be computed in batch
                pwES.append(iisignature.sig(bag, self.order, 2))
            else:
                error_message = 'All time series in a bag must have the same length'
                raise NameError(error_message)

        return [x.mean(0) for x in pwES]


class SignatureTransform(BaseEstimator, TransformerMixin):

    def __init__(self, order):
        if not isinstance(order, int) or order < 1:
            raise NameError('The order must be a positive integer.')
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # get the lengths of all pathwise expected signatures
        lengths = [pwES.shape[0] for pwES in X]
        if len(list(set(lengths))) == 1:
            # if all pathwise expected signatures have the same length, the signatures can be computed in batch
            return iisignature.sig(X, self.order)
        else:
            return [iisignature.sig(item, self.order) for item in X]


if __name__ == "__main__":
    doctest.testmod()
