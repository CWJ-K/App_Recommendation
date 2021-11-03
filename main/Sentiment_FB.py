"""
Created on Thursday  Feb 13, 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Nessya Callista
@Obj   : Use Vader to classify sentiments in FB comments
"""


import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import sent_tokenize
import pandas as pd

#retrieve the reviews that have been scraped and imported to xls file
review_filename = r'C:\Users\R\Desktop\FB.xlsx'
review_sheetname = 'FB'

#put the reviews into dataframe
review_FB = pd.ExcelFile(review_filename)
FB = review_FB.parse(review_sheetname)


### Clean dataframe
#some comments without text, but a graph. Therefore, there is nan in comments
FB.dropna(axis=0, how='any', inplace=True)

#TechTuesdays: is a tag related to technology


comment= []

for word in FB['Comment']:
    if '#techtue' not in word.lower() and ('curry' or 'you' in word.lower()):
        comment.append(word)

df =[]      
df = pd.DataFrame({'Comment':comment},index=range(1,len(comment)+1))





#add new columns to put the sentiment based on the lexicons, VADER and SENTIWORDNET
df['Vader_Label']=''
df['Vader_Compound']=''


sid = SIA() #Create a sentiment intensity analyzer object:

i=0
#Do the Sentiment Analysis (VADER) and save them to the dataframe




for reviews in df['Comment']:
    sen_timent = 0
    revs = sent_tokenize(reviews)
    for rev in revs:
        ss = sid.polarity_scores(rev)
        sen_timent = sen_timent + ss['compound']

    if sen_timent < 0:
        df['Vader_Label'].values[i] = "Negative"
        df['Vader_Compound'].values[i] = sen_timent
        #print(reviews, ' is overall negative ', sen_timent)
    elif sen_timent == 0:
        df['Vader_Label'].values[i] = "Neutral"
        df['Vader_Compound'].values[i] = sen_timent
        #print(reviews, ' is overall neutral')
    else:
        df['Vader_Label'].values[i] = "Positive"
        df['Vader_Compound'].values[i] = sen_timent
        #print(reviews, ' is overall positive ', sen_timent)
    i=i+1

# remove the length of token <3
listcomm_tok = list()
length = list()
for comment in Comment_data:
    text = word_tokenize(comment)
    num = len(text)
    listcomm_tok.append(text)
    length.append(num)
#len(listcomm_tok)
FB['Tok'] = listcomm_tok



#export dataframe to the xls file
df.to_excel("C:/Users/R/Desktop/FB_Sen.xlsx",sheet_name='FB',index = False)

