from utils import process_tweet, lookup  # For pre processing of tweets
import pdb
import math
import nltk
import string
import re   # Regular expression
from nltk.tokenize import TweetTokenizer
import csv # to read the csv file of stop words
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
import csv

import pickle

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from utils import process_tweet, lookup  # For pre processing of tweets
import pdb
import math
import nltk
import string
import re   # Regular expression
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split

import csv
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping



class model():
    def __init__(self):
        #self.tfidfconverter = pickle.load(open('tfidf.pickle', 'rb'))
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        #self.Stop_Words=[]
        #with open('StopWords.csv', 'r') as file:  # Read the stop words from file and store them in the List
        #    reader = csv.reader(file)
        #    i=0
        #    for row in reader:
        #        self.Stop_Words.insert(i,row)
        #        i+=1
        # Recreate the exact same model, including its weights and the optimizer
        
        self.loaded_model = tf.keras.models.load_model('my_model.h5')

        pass

    # A function to pre process the tweets
    def Pre_Process(self,Tweets):    
        Tokenized_Tweets=[]
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True) # Case-Folding, Tokenization
        #for i in range(len(Tweets)-1):
        tweet_tokens = tokenizer.tokenize(Tweets)
        Tokenized_Tweets.extend(tweet_tokens)
        cleaned_tweets = []
        print(Tokenized_Tweets)
        #for i in range (len(Tokenized_Tweets)): # For all tweets
        inner=[]
        for i in range(0,len(Tokenized_Tweets)): # For all the words of one tweet
            check=0
                #for j in range(len(self.Stop_Words)):        # Iterate for all the words of one tweet in Stop Words List
                 #   if(Tokenized_Tweets[i][k]==self.Stop_Words[j][0]): #check if the words of one tweet are in Stop Words List
                  #      check=1 
                   #     break
            if Tokenized_Tweets[i] in string.punctuation:  # Removes punctuation
                check=1
            if not check:        
                inner.append(Tokenized_Tweets[i])
        cleaned_tweets.extend(inner)
        return cleaned_tweets
    
    # convert the tokenized tweets back to string
    def Remove_Stop_Words(self,x):# Combine tokenzied words to strings again after removing stop words
        
        Tweets=[]
        b=''
        i=0
        for tweet in x:
            for word in tweet:
                b=b+word
                b=b+' '
            Tweets.insert(i,b)
            i+=1
        return Tweets


    def predict(self,tweet):
        a = [tweet]
        #a = self.Pre_Process(a)
        
        seq = self.tokenizer.texts_to_sequences(a)
        padded = pad_sequences(seq, maxlen=150)
        pred = self.loaded_model.predict(padded)
        print(type(pred))
        print(pred)
        pred= np.squeeze(pred)
        pred= ('N',pred[0]) if pred[0]>pred[1] else ('R',pred[1])
        
        #print(pred)
        return pred

# if __name__ == "__main__":
#     m = model()
#     m.predict("trump ne michelle obama ke barey mein nasal parastanah jumley kahey")