"""
Created on Friday Feb 17, 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Gandes Aisyaharum, Nessya Callista, Chen Wan Ju
@Obj   : Check the quality of ratings in all reviews and plot AUC
"""
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

file = ['Customers_reviews.xlsx']

# Name of cloumns: App Content_Review Date_Rev ID Reviewer Star_Rating Title platform

# import data: all reviews with platform & app names
data = pd.read_excel(file[0])


#add new columns to store the sentiment based on the star, Sentiword and Vader
data['Star_Label'] = ''
data['Vader_Label'] = ''
data['Vader_Compound'] = ''
data['Sentiword_Label'] = ''
data['Sentiword_Pos'] = ''
data['Sentiword_Neg'] = ''

#Classify the reviews based on the star
i = 0
for star in data['Star_Rating']:              
    if star < 3:
        data['Star_Label'].values[i] = 'Negative'
    elif star == 3:
        data['Star_Label'].values[i] = 'Neutral'
    else:
        data['Star_Label'].values[i] = 'Positive'
    i+=1

    
#Classify the reviews based on the Vader
j=0
for review in data['Content_Review']:
    text = ""
    for line in review:
        text = text+line
        sentences = sent_tokenize(text)
    
    sid = SIA()
    sentiment = 0
    
    # loop the sentences
    for sen in sentences:
        ss = sid.polarity_scores(sen)
        sentiment += ss['compound']
    data['Vader_Compound'].values[j] = sentiment
    if sentiment < 0:
        data['Vader_Label'].values[j] = 'Negative'
    elif sentiment == 0:
        data['Vader_Label'].values[j] = 'Neutral'
    else:
        data['Vader_Label'].values[j] = 'Positive'
    j+=1

#Classify the reviews based on the Sentiword
k=0
for review in data['Content_Review']:
    pos = 0
    neg = 0
    wn_lem = WordNetLemmatizer()
    tokens = nltk.word_tokenize(review)
    for token in tokens:
        lemma = wn_lem.lemmatize(token)
        if len(wn.synsets(lemma)) >0:
            synset = wn.synsets(lemma)[0]
            sent = swn.senti_synset(synset.name())
            sen_pos = pos + sent.pos_score()
            sen_neg = neg + sent.neg_score()
    if sen_pos > sen_neg:
        data['Sentiword_Label'].values[k] = "Positive"
    elif sen_pos==sen_neg:
        data['Sentiword_Label'].values[k] = "Neutral"
    else:
        data['Sentiword_Label'].values[k] = "Negative"
    data['Sentiword_Pos'].values[k]=sen_pos
    data['Sentiword_Neg'].values[k]=sen_neg
    k+=1

data.to_excel(file[0])


star = data['Star_Label'].tolist()
vader = data['Vader_Label'].tolist()
sentiword = data['Sentiword_Label'].tolist()

Star_Vader = ConfusionMatrix(star, vader)
Star_Sentiword = ConfusionMatrix(star, sentiword)
Star_Vader.plot()
plt.show()

Star_Sentiword.plot()
plt.show()
