
# coding: utf-8

# # Task 2: predict language in a document.
# 

# In[1]:


# Python 3.6

import os
import glob
from collections import Counter

import nltk

from src.data_loading.task_2 import load_train, load_test
from src.text_classifier_pipelines.char_ngram.logistic import find_and_train_best_logistic_pipeline
from src.text_classifier_pipelines.char_ngram.multinomial_naive_bayes import find_and_train_best_multinomial_pipeline
from src.evaluation.eval_and_plot import eval_and_plot


# ## First: load data.

# In[2]:


corpus_train = glob.glob(os.path.join(".", "data", "task2", "identification_langue", "corpus_entrainement", "*.txt"))
corpus_test1 = glob.glob(os.path.join(".", "data", "task2", "identification_langue", "corpus_test1", "*.txt"))

print("Train files found:", corpus_train)
print("")
print("Test files found:", corpus_test1)
print("")

X_train, y_train, labels_readable = load_train(corpus_train, sentence_tokenizer=nltk.sent_tokenize)
X_test, y_test = load_test(corpus_test1, labels_readable)

print("'print(len(X_train), len(y_train))':")
print(len(X_train), len(y_train))
print("'print(len(X_test), len(y_test))':")
print(len(X_test), len(y_test))
print("")
print("Do we have a balanced dataset after counting sentences in train, and items in test?")
print("    Train:", Counter(y_train))
print("    Test:", Counter(y_test))
print("Note: Indices are labels' indexes, and counts are their appearance. Labels are:")
print(list(enumerate(labels_readable)))
print("")


# ## Then, create Pipeline() classes for scikit-learn
# 
# Because here, we want to perform those steps in order: 
# 1. Lowercase the text (but keep accents).
# 2. Convert the documents to char ngrams.
# 3. Use the classifier of our choice later on.
# 
# This is done in the `from src.text_classifier_pipelines` local package imported.

# ## Training First Pipeline: TF-IDF Logistic Classifier.
# 
# Cross Validation is performed.

# In[3]:


best_trained_logistic_pipelines = find_and_train_best_logistic_pipeline(X_train, y_train)


# ## Training Second Pipeline: TF-IDF Multinomial Naive Bayes Classifier.
# 
# Cross Validation is performed.

# In[4]:


best_trained_bayes_pipelines = find_and_train_best_multinomial_pipeline(X_train, y_train)


# In[5]:


plot_range = 100

best_trained_pipelines = dict()
best_trained_pipelines.update(best_trained_logistic_pipelines)
best_trained_pipelines.update(best_trained_bayes_pipelines)

# eval_and_plot(best_trained_logistic_pipelines, X_test, y_test, plot_range=plot_range)
# eval_and_plot(best_trained_bayes_pipelines, X_test, y_test, plot_range=plot_range)
eval_and_plot(best_trained_pipelines, X_test, y_test, plot_range=plot_range)

print("")


# In[6]:


print("The final test: classifying on test documents of full-length:")
print("")
for (model_name, model) in best_trained_pipelines.items():
    score = model.score(X_test, y_test)
    print("Score for '{}': {}%".format(model_name, score*100))
print("")

