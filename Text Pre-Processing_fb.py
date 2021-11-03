# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:52:22 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Mirza Miftanula
@Obj   : Text Pre-Processing
"""

import pandas as pd
import os
import numpy as np
import nltk
from nltk import word_tokenize


# Read data
os.chdir(r'C:\Users\R\Desktop')
mydir = os.getcwd()
file = 'FB_Sen.xlsx'
V1_DF_Competitor = pd.read_excel(file,na_values = 'NA',usecols = "A:O",index=0)

len(V1_DF_Competitor)

# Remove the comment which has length <= 3
idxlist = list()
for comment in V1_DF_Competitor['Comment']:
    if len(comment.split())<=3:
        value = 0
    else:
        value = 1
    idxlist.append(value)
    
V1_DF_Competitor['Addvar']=idxlist      
V1_DF_Competitor=V1_DF_Competitor[V1_DF_Competitor['Addvar']==1]

# Extract comment part
Comment_data = V1_DF_Competitor['Comment']

# Perform text pre-processing
# b. Tokenize the word
listcomm_tok = list()
for comment in Comment_data:
    text = word_tokenize(comment)
    listcomm_tok.append(text)
    


# c. Case correction (set all to lower case)
listok_case = list()
for listitem in range(0,len(listcomm_tok)):
    listarray = listcomm_tok[listitem]
    datas = list()
    for word in range(0,len(listarray)):
        data = listarray[word]
        data = data.lower()
        datas.append(data)
    listok_case.append(datas)

print(len(listok_case))
    
# c.1 add the tokenize word in main df
colidx = len(V1_DF_Competitor.columns.values)
V1_DF_Competitor.insert(loc=colidx, column='Tokenize_Comment', value=listok_case)

# d. Clear stand alone punctuation
tes = V1_DF_Competitor['Tokenize_Comment']
words = list()

for item in tes:
    bags = list()
    for token in item:
        if token.isalpha():
            insertword = token
            bags.append(insertword)
    words.append(bags)

## update the column
V1_DF_Competitor['Tokenize_Comment']=words  

# e. Remove stop words from each row
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
## Create additional stopwords
add_sw = ['currys','curry','son','pc','world','pet','hair','smell','u','fridge','today','actually','child','school'
'christmas','animal','dog','cat','hi','really','sorry','dm','apology','store','please','carrie','thank','thanks']

for item in add_sw:
    stop_words.add(item)
tes = V1_DF_Competitor['Tokenize_Comment']
words = list()
for item in range(0,len(tes)):
    bags = list()
    data = tes.iloc[item]
    for word in data:
        if word not in stop_words:
            inonsw = word
            bags.append(inonsw)
    words.append(bags)

# e.1 add the cleaned word in main df
colidx = len(V1_DF_Competitor.columns.values)
V1_DF_Competitor.insert(loc=colidx, column='Tokenize_Comment_RM_SW', value=words)



# f. create position of speech (POS)
token_word = V1_DF_Competitor['Tokenize_Comment_RM_SW']
pos_token_word = list()
for word in token_word:
    tagged = nltk.pos_tag(word)
    pos_token_word.append(tagged)

## f.1 insert the post_token_word into dataframe
colidx = len(V1_DF_Competitor.columns.values)
V1_DF_Competitor.insert(loc=colidx, column='POS_Token_Comment', value=pos_token_word)   

# g. Lemmatize
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wn_lem = WordNetLemmatizer()

# g.1 create function to lemmatize the word based on the POS
def get_pos_tag(words):
    if words.startswith('J'):
        return wn.ADJ
    elif words.startswith('V'):
        return wn.VERB
    elif words.startswith('N'):
        return wn.NOUN
    elif words.startswith('R'):
        return wn.ADV
    else:
        return ''

# g.2 apply lemmatization to each of tokenized word 
sources = V1_DF_Competitor['POS_Token_Comment']
listtest = list()
for i in range(0,len(sources)):
    source = sources.iloc[i]
    listbag = list()
    for j in range(0,len(source)):
        word = source[j][0]
        posit = get_pos_tag(source[j][1])
        if posit != '':
            stem = wn_lem.lemmatize(word,pos = posit)
        else:
            stem = wn_lem.lemmatize(word)
        listbag.append(stem)
    listtest.append(listbag)
## insert the new column
colidx = len(V1_DF_Competitor.columns.values)
V1_DF_Competitor.insert(loc=colidx, column='Tokenize_Comment_LM', value=listtest) 

V1_DF_Competitor_1 = V1_DF_Competitor[(V1_DF_Competitor['App']=='Argos') & (V1_DF_Competitor['Star_Label']=='Negative')]
###### SAVE DATAFRAME
V1_DF_Competitor.to_pickle('V1_DF_Competitor')
