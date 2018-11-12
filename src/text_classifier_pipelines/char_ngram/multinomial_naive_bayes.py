from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.evaluation.cross_validation import get_best_classifier_from_cross_validation
from src.pipeline_steps.to_lower_case import ToLowerCase
from src.pipeline_steps.to_n_gram import ToNGram


def find_and_train_best_multinomial_pipeline(X_train, y_train):
    print("Will start Cross Validation for Naive Bayes (MultinomialNB) Classifiers.")
    print("")

    best_trained_bayes_pipelines = {
        "1-gram Char Multinomial Naive Bayes Classifier": get_best_classifier_from_cross_validation(
            get_multinomial_naive_bayes_hyperparams_grid(ngram=1),
            new_multinomial_naive_bayes_pipeline,
            X_train, y_train,
            name="1-gram", verbose=True
        ),
        "2-gram Char Multinomial Naive Bayes Classifier": get_best_classifier_from_cross_validation(
            get_multinomial_naive_bayes_hyperparams_grid(ngram=2),
            new_multinomial_naive_bayes_pipeline,
            X_train, y_train,
            name="2-gram", verbose=True
        ),
        "3-gram Char Multinomial Naive Bayes Classifier": get_best_classifier_from_cross_validation(
            get_multinomial_naive_bayes_hyperparams_grid(ngram=3),
            new_multinomial_naive_bayes_pipeline,
            X_train, y_train,
            name="3-gram", verbose=True
        )
    }
    return best_trained_bayes_pipelines


def get_multinomial_naive_bayes_hyperparams_grid(ngram):
    return {
        'to_ngram__ngram': [ngram],
        'to_ngram__stride': [1],
        'count_vectorizer__max_df': [0.98],
        'count_vectorizer__min_df': [2],
        'count_vectorizer__max_features': [100000],
        'count_vectorizer__ngram_range': [(1, 1)],
        'count_vectorizer__strip_accents': [None],
        'count_vectorizer__tokenizer': [lambda x: x],
        'count_vectorizer__preprocessor': [None],
        'count_vectorizer__lowercase': [False],
        'tf_idf__norm': [None, 'l1', 'l2'],
        'tf_idf__smooth_idf': [True, False],
        'tf_idf__sublinear_tf': [True, False],
        'naive_bayes_multiclass__alpha': [0.01, 0.1, 1.0]
    }


def new_multinomial_naive_bayes_pipeline():
    return Pipeline([
        ('to_lower_case', ToLowerCase()),
        ('to_ngram', ToNGram()),
        ('count_vectorizer', CountVectorizer()),
        ('tf_idf', TfidfTransformer()),
        ('naive_bayes_multiclass', MultinomialNB()),
    ])
