import numpy as np
from numpy import bool_
from sklearn.base import BaseEstimator, TransformerMixin


class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Convert values to True or False."""

    def __init__(self, true_value="yes", false_value="no"):
        """Initialize BooleanTransformer instance."""
        self.true_value = true_value
        self.false_value = false_value

    def fit(self, X, y=None):
        """Fit the transformer to a dataset."""
        return self

    def transform(self, X, y=None):
        """Transform a dataset."""
        def f(value):
            if type(value) is bool or type(value) is bool_:
                return value
            elif value == self.true_value:
                return True
            elif value == self.false_value:
                return False
            else:
                raise ValueError("Value: {} cannot be mapped to a boolean value.".format(value))

        X = np.vectorize(f)(X)

        return X
