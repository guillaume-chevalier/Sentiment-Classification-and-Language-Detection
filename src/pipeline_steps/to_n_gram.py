from conv import convolved_1d  # pip install conv
from sklearn.base import BaseEstimator, TransformerMixin


class ToNGram(BaseEstimator, TransformerMixin):

    def __init__(self, ngram=3, stride=1):
        self.ngram = ngram
        self.stride = stride

    def get_params(self, deep=True):
        return {
            "ngram": self.ngram,
            "stride": self.stride
        }

    def set_params(self, **params):
        for k, v in params.items():
            self.__setattr__(k, v)
        return self

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        """
        Would convert ["test!", "hey"] to [["tes", "est", "st!"], ["hey"]].
        """
        x = [[
            "".join(ngram)  #
            for ngram in convolved_1d(
                # example: list(convolved_1d("test!", kernel_size=3, stride=1, padding='VALID'))
                # output : [['t', 'e', 's'], ['e', 's', 't'], ['s', 't', '!']]
                sentence, kernel_size=self.ngram, stride=self.stride, padding='VALID'
            )
        ] for sentence in x]

        return x
