from sklearn.base import BaseEstimator, TransformerMixin


class ToLowerCase(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return [[word.lower() for word in sentence] for sentence in x]
