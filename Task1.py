
# coding: utf-8

# # Sentiment analysis and language identification
# 
# This is a project for the Natural Language Processing (NLP) class "IFT-7022 Techniques et applications du traitement automatique de la langue (TALN)" of [Luc Lamontagne](http://www2.ift.ulaval.ca/~lamontagne/). 
# 
# ## Tasks
# 
# There are two tasks to do in this homework: 
# 
# - Task 1: classify positive and negative emotion in a document.
# - Task 2: predict language in a document.
# 
# 
# ## License
# 
# The current code is released under the **BSD-3-Clause license**. See `LICENSE.md`. Copyright 2018 Guillaume Chevalier.
# 
# Note: as this is a university school project, the licences of the imported librairies, datasets, or other assets has not been checked.
# 
# ## Using the provided data
# 
# Here is the structure of the folders for this code to run:
# 
# (Note: all text files have been ignored with grep so as to make the tree shorter, and the tree was captured before creating the src* folder. The data could be downloaded from the course website.)

# In[1]:


get_ipython().system('tree data | grep -v txt | grep -v text')
get_ipython().system('pwd')


# Also see the `requirements.txt` file: 

# In[2]:


get_ipython().system('cat requirements.txt')


# # Task 1: classify positive and negative emotion in a document.

# In[3]:


# Python 3.6

import os
import glob

from src.pipeline_steps.nltk_word_tokenize import NLTKTokenizer
from src.pipeline_steps.to_lower_case import ToLowerCase
from src.pipeline_steps.remove_stop_words import RemoveStopWords
from src.pipeline_steps.keep_open_classes_only import KeepOpenClassesOnly
from src.pipeline_steps.sentiwordnet import SentiWordNetPosNegAttributes
from src.pipeline_steps.porter_stemmer import PorterStemmerStep
from src.pipeline_steps.data_shape_printer import ShapePrinter
from src.data_loading.task_1 import load_all_data_task_1
from src.text_classifier_pipelines.stop_words_open_class_stemmer.pipeline_factory import find_and_train_best_pipelines


# In[4]:


neg_Bk_files = glob.glob(os.path.join(".", "data", "task1", "Book", "neg_Bk", "*.text"))
pos_Bk_files = glob.glob(os.path.join(".", "data", "task1", "Book", "pos_Bk", "*.text"))

X_train, y_train, X_test, y_test = load_all_data_task_1(neg_Bk_files, pos_Bk_files)

print(len(X_train), len(y_train), len(X_test), len(y_test))


# In[5]:


best_trained_pipelines = find_and_train_best_pipelines(X_train, y_train)


# In[6]:


print("The final test: classifying on test documents of full-length:")
print("")
print("Note: the test set was of 20% of full data, which was held-out of cross validation.")
print("")
max_score = 0
max_score_model = ""
for (model_name, model) in best_trained_pipelines.items():
    score = model.score(X_test, y_test) * 100
    if score > max_score: 
        max_score = score
        max_score_model = model_name
    print("Test set score for '{}': {}%".format(model_name, score))
print("")
print("Max score is by '{}': {}%".format(max_score_model, max_score))
print("")

