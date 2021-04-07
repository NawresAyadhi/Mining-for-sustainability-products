import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import os
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')
import joblib
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)
from amazon_analysis.data_preprocessing.text_cleaning import CleanText
from amazon_analysis.data_preprocessing.data_generation import TextCounts
#from data_training.grid_search import grid_vect
from amazon_analysis.data_training.models import mnb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from amazon_analysis.data_preprocessing.utils import ColumnExtractor
from pprint import pprint
from time import time

#best_model = pipeline.fit(df_model.drop('sentiment', axis=1), df_model.sentiment)





def infer_sustainabilty(review):
    loaded_model = joblib.load('output/sus_best_mnb_countvect.pkl')
    tc = TextCounts()
    ct = CleanText()
    new_review = pd.Series(review)
    df_counts = tc.transform(new_review)
    df_clean = ct.transform(new_review)
    df_model = df_counts
    df_model['clean_text'] = df_clean
    scores=loaded_model.predict(df_model).tolist()
    scores_reviews=["mentions sustainability" if x>0.5 else "doesn't mention sustainability" for x in scores]
    print(scores_reviews)
    return scores_reviews