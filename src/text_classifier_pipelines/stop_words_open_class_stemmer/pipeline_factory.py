from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation.cross_validation import get_best_classifier_from_cross_validation
from src.pipeline_steps.keep_open_classes_only import KeepOpenClassesOnly
from src.pipeline_steps.nltk_word_tokenize import NLTKTokenizer
from src.pipeline_steps.porter_stemmer import PorterStemmerStep
from src.pipeline_steps.remove_stop_words import RemoveStopWords
from src.pipeline_steps.sentiwordnet import SentiWordNetPosNegAttributes
from src.pipeline_steps.to_lower_case import ToLowerCase


def find_and_train_best_pipelines(X_train, y_train):
    print("Will start Cross Validation for Logistic Classifiers.")
    print("")

    best_trained_pipelines = {
        "1-gram Char Logistic Classifier": get_best_classifier_from_cross_validation(
            get_generic_hyperparams_grid(True),
            NewLogisticPipelineFunctor(
                tolower=True,
                attributes="remove_stopwords",
                with_pos_neg_attribute=True,
                with_stemming=True,
                logistic_else_bayes=True
            ),
            X_train, y_train,
            name="NAMEE TODO", verbose=True
        )
    }
    return best_trained_pipelines  # TODO: delete above and use the method below.

    best_trained_pipelines = dict()

    for logistic_else_bayes_name, is_logistic_else_bayes in zip(
            ("Logistic Classifier ", "Multinomial Naive Bayes Classifier "), [True, False]):
        for tolower_name, tolower in zip(("with_lowercase ", ""), [True, False]):
            for attributes in ["all_attributes", "remove_stopwords", "keep_only_closed_classes"]:
                for with_pos_neg_attribute_name, with_pos_neg_attribute in zip(("with_pos_neg_attribute ", ""),
                                                                               [True, False]):
                    for with_stemming_name, with_stemming in zip(("with_stemming ", ""), [True, False]):
                        key_name = logistic_else_bayes_name + tolower_name + attributes + " " \
                                   + with_pos_neg_attribute_name + with_stemming_name

                        best_trained_logistic_pipelines[key_name] = get_best_classifier_from_cross_validation(
                            get_generic_hyperparams_grid(is_logistic_else_bayes),
                            NewLogisticPipelineFunctor(
                                tolower=tolower,
                                attributes=attributes,
                                with_pos_neg_attribute=with_pos_neg_attribute,
                                with_stemming=with_stemming,
                                logistic_else_bayes=is_logistic_else_bayes
                            ),
                            X_train, y_train,
                            name=key_name, verbose=True
                        )

    return best_trained_pipelines


def get_generic_hyperparams_grid(is_logistic_else_bayes):
    d = {
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_df': [0.98],
        'count_vect_that_remove_unfrequent_words_and_stopwords__min_df': [2],
        'count_vect_that_remove_unfrequent_words_and_stopwords__max_features': [50000],
        # 'count_vect_that_remove_unfrequent_words_and_stopwords__ngram_range': [(1, 1), (1, 2), (1, 3)],  # TODO:...
        'count_vect_that_remove_unfrequent_words_and_stopwords__ngram_range': [(1, 1)],
        'count_vect_that_remove_unfrequent_words_and_stopwords__strip_accents': [None],
        'count_vect_that_remove_unfrequent_words_and_stopwords__tokenizer': [lambda x: x],
        'count_vect_that_remove_unfrequent_words_and_stopwords__preprocessor': [None],
        'count_vect_that_remove_unfrequent_words_and_stopwords__lowercase': [False],
    }
    if is_logistic_else_bayes:
        # 'logistic_regression__C': [1e-2, 1.0, 1e2, 1e4] # TODO:..
        d['logistic_regression__C'] = [1e4]
    else:
        d['naive_bayes_multiclass__alpha'] = [0.1]  # TODO: [0.01, 0.1, 1.0]
    return d


class NewLogisticPipelineFunctor:
    """This is well salted, sir."""

    def __init__(self, tolower, attributes, with_pos_neg_attribute, with_stemming, logistic_else_bayes):

        self.tolower = tolower
        attributes_selection = {
            "all_attributes": lambda: [],
            "remove_stopwords": lambda: [('remove_stop_words', RemoveStopWords())],
            "keep_only_closed_classes": lambda: [('keep_open_classes_only', KeepOpenClassesOnly())]
        }
        self.attributes_selection = attributes_selection[attributes]()
        self.with_pos_neg_attribute = with_pos_neg_attribute
        self.with_stemming = with_stemming
        self.logistic_else_bayes = logistic_else_bayes

    def __call__(self):
        steps = []
        steps += [('nltk_tokenizer', NLTKTokenizer())]
        if self.tolower:
            steps += [('to_lower_case', ToLowerCase())]
        steps += self.attributes_selection
        if self.with_pos_neg_attribute:
            steps += [('sentiwordnet_attribute_pos_neg_count', SentiWordNetPosNegAttributes())]
        if self.with_stemming:
            steps += [('porter_stemmer', PorterStemmerStep())]
        steps += [('count_vect_that_remove_unfrequent_words_and_stopwords', CountVectorizer())]
        if self.logistic_else_bayes:
            steps += [('logistic_regression', LogisticRegression())]
        else:
            ('naive_bayes_multiclass', MultinomialNB()),
        return Pipeline(steps)
