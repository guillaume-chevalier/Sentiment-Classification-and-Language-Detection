from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import sentiwordnet as swn

nltk.download('sentiwordnet')


class SentiWordNetPosNegAttributes(BaseEstimator, TransformerMixin):
    """
    The strategy here is the lengthen the document with new words
    such as "swnsentipositivexyz" and "swnsentinegativexyz"
    for the Terms Frequency to be computed on that too in later steps, as the
    specification in the homework was to get the "Nombre de mots positifs/négatifs".
    """
    POSITIVE_SENT = "swnsentipositivexyz"
    NEGATIVE_SENT = "swnsentinegativexyz"

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return [self._augment_document_with_sentiwordnet(sentence) for sentence in x]

    @staticmethod
    def _augment_document_with_sentiwordnet(document):
        """
        Example input:
        ["joy", "not", "anger"]
        Example output (notice the reverse/mirror order of the prepends):
        ['swnsentinegative', 'swnsentinegative', 'swnsentipositive', 'joy', 'not', 'anger']
        """
        for word in document:
            try:
                tot_pos = 0
                tot_neg = 0
                for r in swn.senti_synsets(word.lower()):
                    tot_pos += r.pos_score()
                    tot_neg += r.neg_score()
                if not (tot_pos == 0 and tot_neg == 0):
                    if tot_pos > tot_neg:
                        document = [SentiWordNetPosNegAttributes.POSITIVE_SENT] + document
                    else:
                        document = [SentiWordNetPosNegAttributes.NEGATIVE_SENT] + document
            except:
                pass
        return document
