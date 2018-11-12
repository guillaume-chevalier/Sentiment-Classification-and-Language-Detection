"""For debugging."""

from sklearn.base import BaseEstimator, TransformerMixin


class ShapePrinter(BaseEstimator, TransformerMixin):

    def __init__(self, name, **params):
        self.name = name

    def fit(self, x, y=None):
        print("FIT:", self.name, x.shape)
        return self

    def transform(self, x, y=None):
        print("TRANSFORM:", self.name, x.shape)
        return x


class LenPrinter3D(BaseEstimator, TransformerMixin):

    def __init__(self, name, **params):
        self.name = name

    def fit(self, x, y=None):
        print("FIT:", self.name, len(x), len(x[0]), len(x[0][0]))
        return self

    def transform(self, x, y=None):
        print("TRANSFORM:", self.name, len(x), len(x[0]), len(x[0][0]))
        return x
