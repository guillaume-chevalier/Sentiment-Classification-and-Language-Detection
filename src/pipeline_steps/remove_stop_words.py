import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class RemoveStopWords(BaseEstimator, TransformerMixin):

    def __init__(self, **k):
        self.stop_words = [
            sw.lower() for sw in (
                    set(sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS) &
                    set(stopwords.words('english'))
            )
        ]

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return [[word for word in sentence if word.lower() not in self.stop_words] for sentence in x]
