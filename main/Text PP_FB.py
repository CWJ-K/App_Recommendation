# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:52:22 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Mirza Miftanula, Chen Wan Ju
@Obj   : Text Pre-Processing (Adjusted for FB)
"""
#########ab=pd.Series(listitem, dtype=object).fillna('nan')
##########FB.to_excel("C:/Users/R/Desktop/test.xlsx",sheet_name='FB')
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
# Read data
file = r'C:\Users\R\Desktop\FB_Sen.xlsx'
review_sheetname = 'FB'
FB_DF = pd.ExcelFile(file)
FB = FB_DF.parse(review_sheetname)

# Extract comment part
Comment_data = FB['Comment']



# Perform text pre-processing
# a. Tokenize the word
listcomm_tok = list()
for comment in Comment_data:
    text = word_tokenize(comment)
    listcomm_tok.append(text)
#len(listcomm_tok)
Comment_data = FB['Comment']
#len(Comment_data)







# b. Case correction (set all to lower case)
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

# b.1 add the tokenize word in main df
colidx = len(FB.columns.values)
FB.insert(loc=colidx, column='Tokenize_Comment', value=listok_case)

# c. Clear stand alone punctuation              
tes = FB['Tokenize_Comment']
words = list()
for item in range(0,len(tes)):    
    bags = list()
    data = tes[item]
    for word in data:
        if word.isalpha():
            insertword = word
            bags.append(insertword)
    words.append(bags)


## update the column
FB.update(pd.DataFrame({'Tokenize_Comment':words}))


# d. Remove stop words from each row
stop_words = set(stopwords.words('english'))
## Create additional stopwords
add_sw = ['currys']
for item in add_sw:
    stop_words.add(item)
tes = FB['Tokenize_Comment']
words = list()
for item in range(0,len(tes)):
    bags = list()
    data = tes[item]
    for word in data:
        if word not in stop_words:
            inonsw = word
            bags.append(inonsw)
    words.append(bags)


# d.1 add the cleaned word in main df
colidx = len(FB.columns.values)
FB.insert(loc=colidx, column='Tokenize_Comment_RM_SW', value=words)



# e. create position of speech (POS)
token_word = FB['Tokenize_Comment_RM_SW']
pos_token_word = list()
for word in token_word:
    tagged = nltk.pos_tag(word)
    pos_token_word.append(tagged)

## e.1 insert the post_token_word into dataframe
colidx = len(FB.columns.values)
FB.insert(loc=colidx, column='POS_Token_Comment', value=pos_token_word)   

# f. Lemmatize

wn_lem = WordNetLemmatizer()

# f.1 create function to lemmatize the word based on the POS
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

# f.2 apply lemmatization to each of tokenized word 
sources = FB['POS_Token_Comment']
listtest = list()
for i in range(0,len(sources)):
    source = sources[i]
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
colidx = len(FB.columns.values)
FB.insert(loc=colidx, column='Tokenize_Comment_LM', value=listtest) 

###### SAVE DATAFRAME
FB.to_pickle('FB')
