from nltk import PorterStemmer as ps
from sklearn.base import BaseEstimator, TransformerMixin

class PorterStemmerStep(BaseEstimator, TransformerMixin):
    """
    Example usage:

    s = "bacon - I like that, ouch wow damn hey bye okay oh m-hm huh"
    PorterStemmerStep().fit_transform(X=NLTKTokenizer().fit_transform([s])[0])
    """

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        stemmer = ps()

        transformed_x = []
        for tokenized_sample in x:
            transformed_sample = []
            for word in tokenized_sample:
                if type(word) != str:
                    print(type(word), len(word))
                transformed_sample.append(
                    stemmer.stem(word)
                )
            transformed_x.append(transformed_sample)

        return transformed_x
