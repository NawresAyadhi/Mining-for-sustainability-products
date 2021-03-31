import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
import re
import emoji
import pandas as pd 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)    
    
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X


def annotate_positive_negative(df):
    return df['sentiment'].apply(lambda x: 0 if int(float(x[0:3])) < 3 else 2 if int(float(x[0:3])) > 3 else 1)


_vocab=["Died after", "doesn’t work", "after", "life","cycle",
        "useless", "lifespan", "life cycle", "died", "longer", "battery", "use", "reuse", "energy","burn",
        "cutting cord", "cable bill", "cutting cable", "cord bill", "last", "environment",
        "unusable", "crashed", "longer life", "change battery", "expensive",
        "cheaper", "only lasted", "burn out", "died less than", "waste",
        "replace battery ever", "change battery every", "last long", "come back life", 
        "died within", "drains battery", "stopped working", "reuse old", "recycling", 
        "eats battery", "good environment", "save money", "pay cable", "cable free",
        "plastic free", "product price", "quality", "easy use", "durable",
        "working", "tired paying bills", "reuse", "resuscitate", "power consumption",
        "energy efficient", "saving more", "lower cable bill", "more cable","cable cutter", "durability",
        "remote broke", "waste money","energy","pollution","sustainable","quality life",
        "carbon","transport","ecological", "reduce energy", "power efficient", "energy efficiency", 
        "environmental impact", "eco-friendly", "single-use", "use once", "garbage", "eco",
        "landfill", "materials use", "use plastic", "easy set up", "tossed garbage", "garbage disposal", 
        "keep dying", "low price", "durability", "after weeks", "after month","after months", "after months",
        "broke after", "cable bill", "week", "Reduce energy consumption" , "Packaging made wood fiber-based materials", "fiber"]

vocab_tmp=[ y for y in (x.split(' ') for x in _vocab) ]  
vocab=[val.lower() for sublist in vocab_tmp for val in sublist]
def subset(x):
    x=x.lower()
    lst=x.split(' ')
    set1 = set(lst) 
    for e in vocab:
        set2 = set(e) 
        if set1.intersection(set2): 
            return True 
        else: 
            return False
def annotate_sustainable_not(df):
    return df['review'].apply(lambda x: 1 if subset(x) else 0)



