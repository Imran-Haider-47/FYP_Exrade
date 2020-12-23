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

import sklearn
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
import csv

import pickle

class model():
    def __init__(self):
        self.tfidfconverter = pickle.load(open('tfidf.pickle', 'rb'))

        self.Stop_Words=[]
        with open('StopWords.csv', 'r') as file:  # Read the stop words from file and store them in the List
            reader = csv.reader(file)
            i=0
            for row in reader:
                self.Stop_Words.insert(i,row)
                i+=1

        self.loaded_model = pickle.load(open('NeuralNetworks.sav', 'rb'))
        pass

    # A function to pre process the tweets
    def Pre_Process(self,Tweets):    
        Tokenized_Tweets=[[]]
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True) # Case-Folding, Tokenization
        for i in range(len(Tweets)-1):
            tweet_tokens = tokenizer.tokenize(Tweets[i])
            Tokenized_Tweets.insert(i,tweet_tokens)
        cleaned_tweets = []
        for i in range (len(Tokenized_Tweets)-1): # For all tweets
            inner=[]
            for k in range(0,len(Tokenized_Tweets[i])): # For all the words of one tweet
                check=0
                for j in range(len(self.Stop_Words)):        # Iterate for all the words of one tweet in Stop Words List
                    if(Tokenized_Tweets[i][k]==self.Stop_Words[j][0]): #check if the words of one tweet are in Stop Words List
                        check=1 
                        break
                if Tokenized_Tweets[i][k] in string.punctuation:  # Removes punctuation
                    check=1
                if not check:        
                    inner.append(Tokenized_Tweets[i][k])
            cleaned_tweets.append(inner)
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
        a = [tweet,[]]
        a = self.Pre_Process(a)
        a = self.Remove_Stop_Words(a)

        x = self.tfidfconverter.transform(a).toarray()
        ypred = self.loaded_model.predict(x)
        return ypred