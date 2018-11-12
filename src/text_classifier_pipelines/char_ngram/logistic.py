from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils import identity
from src.evaluation.cross_validation import get_best_classifier_from_cross_validation
from src.pipeline_steps.to_lower_case import ToLowerCase
from src.pipeline_steps.to_n_gram import ToNGram


def find_and_train_best_logistic_pipeline(X_train, y_train):
    print("Will start Cross Validation for Logistic Classifiers.")
    print("")

    best_trained_logistic_pipelines = {
        "1-gram Char Logistic Classifier": get_best_classifier_from_cross_validation(
            get_logistic_hyperparams_grid(ngram=1),
            new_logistic_pipeline,
            X_train, y_train,
            name="1-gram", verbose=True
        ),
        "2-gram Char Logistic Classifier": get_best_classifier_from_cross_validation(
            get_logistic_hyperparams_grid(ngram=2),
            new_logistic_pipeline,
            X_train, y_train,
            name="2-gram", verbose=True
        ),
        "3-gram Char Logistic Classifier": get_best_classifier_from_cross_validation(
            get_logistic_hyperparams_grid(ngram=3),
            new_logistic_pipeline,
            X_train, y_train,
            name="3-gram", verbose=True
        )
    }
    return best_trained_logistic_pipelines


def get_logistic_hyperparams_grid(ngram):
    return {
        'to_ngram__ngram': [ngram],
        'to_ngram__stride': [1],
        'count_vectorizer__max_df': [0.98],
        'count_vectorizer__min_df': [1],
        'count_vectorizer__max_features': [100000],
        'count_vectorizer__ngram_range': [(1, 1)],
        'count_vectorizer__strip_accents': [None],
        'count_vectorizer__tokenizer': [identity],
        'count_vectorizer__preprocessor': [None],
        'count_vectorizer__lowercase': [False],
        'tf_idf__norm': [None, 'l1', 'l2'],
        'tf_idf__smooth_idf': [True, False],
        'tf_idf__sublinear_tf': [True, False],
        'logistic_regression__C': [1e3, 1e4, 1e5],
    }


def new_logistic_pipeline():
    return Pipeline([
        ('to_lower_case', ToLowerCase()),
        ('to_ngram', ToNGram()),
        ('count_vectorizer', CountVectorizer()),
        ('tf_idf', TfidfTransformer()),
        ('logistic_regression', LogisticRegression()),
    ])
