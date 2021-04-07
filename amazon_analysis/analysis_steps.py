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
from amazon_analysis.data_preprocessing.text_cleaning import CleanText, annotate_sustainable_not
from amazon_analysis.data_preprocessing.data_generation import TextCounts
#from data_training.grid_search import grid_vect
from amazon_analysis.data_training.models import mnb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from amazon_analysis.data_preprocessing.utils import ColumnExtractor
from pprint import pprint
from time import time


def grid_vect(clf, parameters_clf, X_train, X_test, parameters_text=None, vect=None,  y_train=None, y_test=None):
    
    textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags'
                      ,'count_mentions','count_urls','count_words']
    
    features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                 , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                                , n_jobs=-1)    
    pipeline = Pipeline([
        ('features', features)
        , ('clf', clf)
    ])
    
    # Join the parameters dictionaries together
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)    
    # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)    
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()    
    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
                        
    return grid_search

def read_data(path):
    df=pd.read_csv(path)
    return df


def encode_target(df):
    df_sentiment=annotate_sustainable_not(df)
    return df_sentiment

def generate_new_columns(df, df_sentiment):
    tc = TextCounts()
    df_eda = tc.fit_transform(df.review)
    df_eda['sustainability'] = df_sentiment
    return df_eda

def clean_text(df):
    ct = CleanText()
    sr_clean = ct.fit_transform(df.review)
    empty_clean = sr_clean == ''
    print('{} records have no words left after text cleaning'.format(sr_clean[empty_clean].count()))
    sr_clean.loc[empty_clean] = '[no_text]'
    cv = CountVectorizer()
    bow = cv.fit_transform(sr_clean)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
    return sr_clean

    

def merge_all_in_single_dataframe(df_eda, sr_clean):
    df_model = df_eda
    df_model['clean_text'] = sr_clean
    df_model.columns.tolist()
    return df_model

def split_data(df_model, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(df_model.drop('sustainability', axis=1), df_model.sustainability, test_size=test_size, random_state=37)
    return X_train, X_test, y_train, y_test

# Parameter grid settings for the vectorizers (Count and TFIDF)
parameters_vect = {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
    'features__pipe__vect__min_df': (1,2)
}
# Parameter grid settings for MultinomialNB
parameters_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
}


def train_model_plus_grid_search(X_train, X_test, y_train, y_test):
    import os 
    current_directory = os.getcwd()
    print(current_directory)
    if not os.path.exists(current_directory+"/output"):
        os.makedirs(current_directory+"/output")
    countvect = CountVectorizer()
    best_mnb_countvect = grid_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=countvect, y_train=y_train, y_test=y_test)
    joblib.dump(best_mnb_countvect, 'output/sus_best_mnb_countvect.pkl')

    tfidfvect = TfidfVectorizer()
    best_mnb_tfidf = grid_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=tfidfvect, y_train=y_train, y_test=y_test)
    joblib.dump(best_mnb_tfidf, 'output/sus_best_mnb_tfidf.pkl')