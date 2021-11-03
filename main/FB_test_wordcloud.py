"""
Created on Sun March 17, 2019

@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Nessya Callista
@Obj   : Topic / Feature Extraction

"""

import nltk
import spacy
import numpy as np
import matplotlib.pyplot as plt
import pprint
spacy.load('en_core_web_sm')
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
parser = English()

#nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
#print (en_stop)

#create function to clean the reviews from stopwords, puntuaction, the words that have less affection to create the topic
#get a lemma for each token

stopwords = ['currys','curry','son','pc','world','pet','hair','smell','u','fridge','today','actually','child','school'
'christmas','animal','dog','cat','hi','really','sorry','dm','apology','store','please','carrie','thank','thanks']

def prepare_text_for_process(text):
    wn_lem = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [wn_lem.lemmatize(token) for token in tokens]
    for word in stopwords:
        tokens = [token for token in tokens if token.lower() != word]
    return tokens




review_filename = r'C:\Users\R\Desktop\FB_Sen.xlsx'
review_sheetname = 'FB'
review_FB = pd.ExcelFile(review_filename)
f = review_FB.parse(review_sheetname)

#set the starting value used in generating random number
import random
random.seed(3)

text_neg = []
text_pos = []
text_neu = []

neg_tweets=[]
pos_tweets=[]
neu_tweets=[]

#read the file of the reviews
i=0
j=0
k=0
l=0
for x in f['Vader_Label']:
    if (f['Vader_Label'][l] == "Negative"):
        neg_tweets.append(f['Comment'][l])
        #print("Tweet : "+str(neg_tweets[i])+" , Category : "+str(f['Vader_Label'][l]))
        #i=i+1
    elif (f['Vader_Label'][l] == "Positive"):
        pos_tweets.append(f['Comment'][l])
        #print("Tweet : "+str(pos_tweets[j])+" , Category : "+str(f['Vader_Label'][l]))
        #j=j+1
    else:
        neu_tweets.append(f['Comment'][l])
        #print("Tweet : "+str(neu_tweets[k])+" , Category : "+str(f['Vader_Label'][l]))
        #k=k+1
    l=l+1

for line in neg_tweets:
    #print(line)
    tokens = prepare_text_for_process(line)
    #print("tokens", tokens)
    #if random.random() > .99:
    #print("after random", tokens, "\n")
    text_neg.append(tokens)

for linepos in pos_tweets:
    #print(line)
    tokens = prepare_text_for_process(linepos)
    #print("tokens", tokens)
    #if random.random() > .99:
    #print("after random", tokens, "\n")
    text_pos.append(tokens)

#print(text_neg)

#display the wordcloud from all reviews
neg_cleans = [' '.join(x) for x in text_neg] 
df_FB_neg= pd.DataFrame()
nc = [' '.join(neg_cleans)]
print("Word Cloud - All Neg FB comments")
wnc = WordCloud(background_color='white').generate(str(nc))
print(rev_cleans)
print(rev_cleans[4])
plt.imshow(wnc, interpolation='bilinear')
plt.axis("off")
plt.show()

# Positice
pos_cleans = [' '.join(x) for x in text_pos] 
df_FB_pos= pd.DataFrame()
pc = [' '.join(pos_cleans)]
print("Word Cloud - All Positive FB comments")
wpc = WordCloud(background_color='white').generate(str(pc))
#print(rev_cleans)
#print(rev_cleans[4])
plt.imshow(wpc, interpolation='bilinear')
plt.axis("off")
plt.show()

