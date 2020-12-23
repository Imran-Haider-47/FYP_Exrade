
# coding: utf-8

# In[7]:


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import process_tweet, lookup  # For pre processing of tweets
import pdb
import math
import nltk
import string
import re   # Regular expression
from nltk.tokenize import TweetTokenizer


# In[8]:


import csv
Stop_Words=[]
with open('StopWords.csv', 'r') as file:  # Read the stop words from file and store them in the List
    reader = csv.reader(file)
    i=0
    for row in reader:
        Stop_Words.insert(i,row)
        i+=1
file.close()


# In[9]:


# A function to pre process the tweets
def Pre_Process(Tweets):
    
    # remove stock market tickers like $GE

    #tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"

     #  tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks

 #   tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # remove hashtags

    # only removing the hash # sign from the word

  #  tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    Tokenized_Tweets=[[]]
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True) # Case-Folding, Tokenization
    for i in range(len(Tweets)-1):
        tweet_tokens = tokenizer.tokenize(Tweets[i])
        Tokenized_Tweets.insert(i,tweet_tokens)
    #print("Number of Tokenized Tweets = "+str(len(Tokenized_Tweets)))
    #print(Tokenized_Tweets)
    #print("len(Tokenized)",len(Tokenized_Tweets))
    cleaned_tweets = []
    for i in range (len(Tokenized_Tweets)): # For all tweets
        inner=[]
        for k in range(0,len(Tokenized_Tweets[i])): # For all the words of one tweet
            check=0
            for j in range(len(Stop_Words)):        # Iterate for all the words of one tweet in Stop Words List
                if(Tokenized_Tweets[i][k]==Stop_Words[j][0]): #check if the words of one tweet are in Stop Words List
                    check=1 
                    break
            if Tokenized_Tweets[i][k] in string.punctuation:  # Removes punctuation
                check=1
            if not check:        
                inner.append(Tokenized_Tweets[i][k])
        cleaned_tweets.append(inner)
    return cleaned_tweets
    
    


# In[10]:


import xlrd       # For reading excel file
from array import *   
loc=("C:\\Users\\Mr.Wick\\Videos\\Fall2020\\FYP\\dataset.xlsx") # Loading dataset 
wb = xlrd.open_workbook(loc)   
sheet = wb.sheet_by_index(0)



TWEETS=[]   # a list to store tweets
labels=[]   # A list to store labels

rows=sheet.nrows

for i in range(1,rows):
    labels.insert(i,sheet.cell_value(i,1))
    TWEETS.insert(i,sheet.cell_value(i,0))
Log_Prior=0
Racial=0
Non_Racial=0
for i in range(len(labels)-1):
    if(labels[i]=='R'):
        Racial+=1
    else:
        Non_Racial+=1
print("No. of Racial Tweets : ",Racial)
print("No. of Non Racial Tweets : ",Non_Racial)
Log_Prior=Racial/Non_Racial
print("Log Prior = ", Log_Prior)
    
cleaned_tweets=Pre_Process(TWEETS)
#print(cleaned_tweets)


# In[11]:


from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test= train_test_split(cleaned_tweets,labels,test_size=0.2)


# In[12]:


i=0
dic=[[]]
label_=Y_Train
positive=0
Negative=0
for tweet in X_Train:
    for word in tweet:
        if label_[i]=='R':
            tup=[word,'R']
            dic.insert(i,tup)
        else:
            tup=[word,'N']
            dic.insert(i,tup)
    i+=1
print("len of dic= ",len(dic))
print("i = ",i)
i=0
j=0

Dictionary=[[]]
def in_dictionary(word):
    for i in range(len(Dictionary)-1):
        if word==Dictionary[i][0]:
            return True
    else:
        return False
i=0
length=0

def Return_Index(word):
    i=0
    for i in range (len(Dictionary)-1):
        if word==Dictionary[i][0]:
            return i

for i in range (len(dic)-1):
    if (in_dictionary(dic[i][0])==True) and (dic[i][1]=='R'):
        length=Return_Index(dic[i][0])
        Dictionary[length][1]=Dictionary[length][1]+1
    elif (in_dictionary(dic[i][0])==True) and (dic[i][1]=='N'):
        length=Return_Index(dic[i][0])
        Dictionary[length][2]=Dictionary[length][2]+1
    else:
        if (dic[i][1]=='N'):
            tuples=[dic[i][0],0,1]
        else:
            tuples=[dic[i][0],1,0]
        Dictionary.insert(len(Dictionary)-1,tuples)
#print(Dictionary)

i=0
No_unique_words=len(Dictionary)-1

Sum_positive_freq=0
Sum_negative_freq=0
for i in range(len(Dictionary)-1):
    Sum_positive_freq=Sum_positive_freq+Dictionary[i][1]
    Sum_negative_freq=Sum_negative_freq+Dictionary[i][2]
#print(Sum_positive_freq)
#print(Sum_negative_freq)
#print(No_unique_words)


# In[13]:


# Calculating Probabilties of each unique word in each class

i=0
Dict_of_Probs=[[]]
for i in range(len(Dictionary)-1):
    
    pos_prob=(Dictionary[i][1]+1)/(Sum_positive_freq+No_unique_words)
    neg_prob=(Dictionary[i][2]+1)/(Sum_negative_freq+No_unique_words)
    Lambda=math.log(pos_prob/neg_prob)
    row=[Dictionary[i][0],pos_prob,neg_prob,Lambda]
    Dict_of_Probs.insert(i,row)
    
#print(Dict_of_Probs)
def in_prob_dic(word):
    for i in range(len(Dict_of_Probs)-1):
        if (word==Dict_of_Probs[i][0]):
            return Dict_of_Probs[i][3]
    return 0
        
        



def Score():
    #def Evaluate(test_twee):
    log_likelihood=0
    likelihood=0
    
    positive_count=0 # to count the number of predicted correct tweets
    
    j=0
    for tweetss in X_Test:
        if(tweetss== []):
            continue
        for i in range(len(tweetss)-1):
            likelihood=in_prob_dic(tweetss[i])
            log_likelihood=log_likelihood+likelihood
        log_likelihood=Log_Prior+log_likelihood

        if (log_likelihood>0):
            if(Y_Test[j]=='R'):
                positive_count+=1
        elif (log_likelihood<0):
            if(Y_Test[j]=='N'):
                positive_count+=1
        else:
            Print("Error in classifying the tweet")
        log_likelihood=0
        likelihood=0
        j+=1
    score=positive_count/len(X_Test)
    return score


# In[14]:


print("Accuracy =", Score())


# In[15]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


def Remove_Stop_Words(x):# Combine tokenzied words to strings again after removing stop words

    Tweets=[]
    a=' '
    i=0
    for tweet in x:
        

        for word in tweet:
            a=a+word
            a=a+' '
        Tweets.insert(i,a)
        i+=1
    return Tweets


# In[17]:


t= Remove_Stop_Words(cleaned_tweets)


# In[59]:


Labels=[]
i=0
for label in (labels):
    if(label=='R'):
        Labels.insert(i,1)
    else:
        Labels.insert(i,0)
    i+=1
len(Labels)


# In[62]:


#Converting the tweets to the TF-IDFs
tfidfconverter = TfidfVectorizer()
X = tfidfconverter.fit_transform(t).toarray()
print(len(X))


# In[51]:


# Splitting the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)




# Random Forest Classifier
classifier2 = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)

#Save neural network model
import pickle
filename='RandomForest.sav'
pickle.dump(classifier3, open(filename, 'wb'))




# Random Forest Classifier
classifier2 = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[56]:


# Neural Network 
import sklearn
from sklearn.neural_network import MLPClassifier
classifier3=MLPClassifier(alpha=1, max_iter=2000)
classifier3.fit(X_train, y_train)
Y_pred = classifier3.predict(X_test)
print(confusion_matrix(y_test,Y_pred))
print(classification_report(y_test,Y_pred))
print(accuracy_score(y_test, Y_pred))


# In[49]:


#Save neural network model
import pickle
filename='finalized.sav'
pickle.dump(classifier3, open(filename, 'wb'))


# In[52]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)





