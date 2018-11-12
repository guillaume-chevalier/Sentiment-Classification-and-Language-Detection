from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class NLTKTokenizer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return [word_tokenize(sentence) for sentence in x]
