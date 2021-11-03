# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:52:22 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Mirza Miftanula
@Obj   : Topic Extraction
"""

import spacy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import nltk
from nltk.collocations import *
from spacy.lang.en import English
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from os import path
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import pickle

random.seed(3)
# Change working directory
os.chdir(r'C:\Users\R\Desktop')
mydir = os.getcwd()

# Import list of lexicon correlated with negative and positive opinion 
    # (Source paper : Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
    #  Proceedings of the ACM SIGKDD International Conference on Knowledge 
    #  Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
    #  Washington, USA,) ..... 
    #  Website : http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

# Negative Opinion
# Read txt data
with open("negative-words.txt", "r",encoding="utf8", errors='ignore') as f:
    neg_opinion=f.read().splitlines()

# Select relevant values
neg_opinion = neg_opinion[neg_opinion.index('abnormal'):len(neg_opinion)]

# Positive Opinion
# Read txt data
with open("positive-words.txt", "r") as f:
    pos_opinion=f.read().splitlines()

# Select relevant values
pos_opinion = pos_opinion[pos_opinion.index('a+'):len(pos_opinion)]


## Retrieve the original dataframe
Original_DF =  pd.read_pickle('V1_DF_Competitor')
Original_DF['Social_Media'] ='FB'
### CLUSTERING ####
###################

## A. Create list to represent the way the code is processed
listSocial_Media = list(set(Original_DF['Social_Media']))
vaderlabel = list(set(Original_DF['Vader_Label']))
# B. Create DF to store the final value of cluster
Sumtable = pd.DataFrame(columns = ["Social_Media","Category","NbofK","Cluster_ID","Mean","Median","Stdev"] ) 
row_id=0
# C. Do Loop Process
for SocialMediaid in listSocial_Media:
    for level in vaderlabel:
       # C.1 Create subset dataframe based on 2 criteria (SocialMediaid,level) 
           DF_Main = Original_DF[(Original_DF['Social_Media']==SocialMediaid) & (Original_DF['Vader_Label']==level)]
           df = [x for w in DF_Main['Tokenize_Comment_LM'] for x in w]
           df_unique = list(set(df))
       # C.2 Join the tokenize words into a complete sentence.
           join_words = [' '.join(x) for x in DF_Main['Tokenize_Comment_LM']]
 
       # C.3 Insert join_words in initial df
           DF_Main["Clean_Reviews"]=join_words

       # C.4 Create empty dataframe to store cluster_id
           DF_Main["Cluster_ID"] = ""

       # C.5 Create TF-IDF Matrix
           vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
           X = vectorizer.fit_transform(join_words)
           tfidf_matrix = X.todense()
           df_tfidf = pd.DataFrame(tfidf_matrix,columns=vectorizer.get_feature_names())

       # C.6 Count no of features which appear in documents
           worddict = vectorizer.get_feature_names()

       # C.7 Creating Cluster using k-means
           for nb_cl in range(2,21):
               # C.7.1 Create the number of cluster
                   true_k = nb_cl
               # C.7.2 Classify the reviews into the clusters
                   model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1, random_state=42)
                   model.fit(X)
               # C.7.3 Insert the Cluster Value into dataframe
                   i = 0
                   for p in DF_Main["Clean_Reviews"]:
                       Y = vectorizer.transform([p])
                       prediction = model.predict(Y)
                       DF_Main["Cluster_ID"].iloc[i] = prediction[0]
                       i+=1
               # C.7.4 Calculate Cosine_Similarity for each cluster
                   for clid in list(set(DF_Main["Cluster_ID"])): 
                       # C.7.4.1 Create referrence dataframe
                           df_pairwise = DF_Main[DF_Main["Cluster_ID"]==clid]
                           if len(df_pairwise)>1:
                           # C.7.4.2 Create the Document Term Matrix
                               count_vectorizer = CountVectorizer(stop_words='english')
                               count_vectorizer = CountVectorizer()
                               sparse_matrix = count_vectorizer.fit_transform(df_pairwise['Clean_Reviews'])
                               doc_term_matrix = sparse_matrix.todense()
                               df = pd.DataFrame(doc_term_matrix,columns=count_vectorizer.get_feature_names())
                               # C.7.4.3 Compute Cosine Similarity
                               cos_matrix = cosine_similarity(df, df)
        
                           # C.7.4.4 Store the data in the df        
                               datalist = list()
                               for i in range(0,len(cos_matrix)-1):
                                   for j in range(i+1,len(cos_matrix)):
                                       datalist.append(cos_matrix[i][j])
                               rec1 = pd.DataFrame(datalist)
                               rec1 = rec1.replace(0,np.NaN)
                               Sumtable.loc[row_id] = [SocialMediaid,level,nb_cl,clid,rec1.mean()[0],rec1.median()[0],rec1.std()[0]]
                               row_id +=1
                           else :
                               Sumtable.loc[row_id] = [SocialMediaid,level,nb_cl,clid,0,0,0]
                               row_id +=1

                   print("SocialMediaid : "+str(SocialMediaid)+" , Category : "+str(level)+", Iteration : " + str(nb_cl))

## Save SumTable
Sumtable.to_pickle('SumTable')

Sumtable.to_excel("C:/Users/R/Desktop/FB.xlsx",sheet_name='FB',index = False)
#Sumtable =  pd.read_pickle('Sumtable')

# D. FORMING FINAL CLUSTER BASED ON PREDETERMINED K
for SocialMediaid in listSocial_Media:
    for level in ('Positive','Negative'):
        DF_Main = Original_DF[(Original_DF['Social_Media']==SocialMediaid) & (Original_DF['Vader_Label']==level)]
        df = [x for w in DF_Main['Tokenize_Comment_LM'] for x in w]
        df_unique = list(set(df))
        join_words = [' '.join(x) for x in DF_Main['Tokenize_Comment_LM']]
        DF_Main["Clean_Reviews"]=join_words
        DF_Main["Cluster_ID"] = ""
        vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
        X = vectorizer.fit_transform(join_words)
        tfidf_matrix = X.todense()
        df_tfidf = pd.DataFrame(tfidf_matrix,columns=vectorizer.get_feature_names())
        worddict = vectorizer.get_feature_names()
        if SocialMediaid=='FB' and level=='Positive':
            fclust= 7
        elif SocialMediaid=='FB' and level=='Negative':
            fclust = 5
        final_model = KMeans(n_clusters=fclust, init='k-means++', max_iter=1000, n_init=1, random_state=42)
        final_model.fit(X)
        i = 0
        for p in DF_Main["Clean_Reviews"]:
            Y = vectorizer.transform([p])
            prediction = final_model.predict(Y)
            DF_Main["Cluster_ID"].iloc[i] = prediction[0]
            i+=1
        filename = SocialMediaid+level
        DF_Main.to_pickle(filename)  






#cek_data = pd.read_pickle('ArgosPositive')
#cek_data_2 = cek_data[cek_data['Cluster_ID']==5]

### TOPIC_EXTRACTION ####
#########################
# A. Build the function to filter the bigram
def BG_F2(ngram,val):
  # A.1 Create Acceptable Order of Bigram (e.g Verb-Adverb,Verb-Noun,Adjective-Noun,Noun-Noun)
    first_gram_1 = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBD', 'VBZ')
    first_gram_2 = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_gram_1 = ('RB','RBR','RBS','WRB', 'NN', 'NNS', 'NNP', 'NNPS','NN', 'NNS', 'NNP', 'NNPS')
    second_gram_2 = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    # A.2 Remove bigram which in the opinion lexicon (positive)
    if val == 'Positive' and tags[0][0] not in pos_opinion:
        # A.2.1 Create the condition if the first bigram is verb
        if tags[0][1] in first_gram_1:
            if tags[0][0] != tags[1][1] and tags[1][1] in second_gram_1 :
                return True
            else :
                return False
        # A.2.2 Create the condition if the first bigram is not verb       
        elif tags[0][1] in first_gram_2 :    
            if tags[0][0] != tags[1][1] and tags[1][1] in second_gram_2 :
                return True
            else :
                return False
        else:
            return False
    # A.3 Remove bigram which in the opinion lexicon (negative)
    if val == 'Negative' and tags[0][0] not in neg_opinion:
        # A.3.1 Create the condition if the first bigram is verb
        if tags[0][1] in first_gram_1:
            if tags[0][0] != tags[1][1] and tags[1][1] in second_gram_1 :
                return True
            else :
                return False
        # A.3.2 Create the condition if the first bigram is not verb       
        elif tags[0][1] in first_gram_2 :    
            if tags[0][0] != tags[1][1] and tags[1][1] in second_gram_2 :
                return True
            else :
                return False
        else:
            return False
    else:
        return False
 

## C. Create dataframe to store the summary bigram in each cluste
Bigram_Summary =pd.DataFrame(columns=["Social_Media","Category","Cluster_ID","Similarity_Score","Top_5_Bigram"]) 

### D.Loop process to store the bigram summary in the dataframe
row_id = 0
for SocialMediaid in listSocial_Media:
    for level in ('Positive','Negative'):
        name=SocialMediaid+level
        DF_Main = pd.read_pickle(name)
        for cluster in list(set(DF_Main["Cluster_ID"])):
            # D.1 Extract words from each cluster
              df_token = DF_Main['Tokenize_Comment_LM'][DF_Main['Cluster_ID']==cluster]
              tokenize_words = [w for item in df_token for w in item]

            # D.2 Extract the bigram for each cluster using and store top 5 bigram 
              bigram = nltk.collocations.BigramCollocationFinder.from_words(tokenize_words)
              bigram_freq= bigram.ngram_fd.items()
              df_bigram = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)


            # D.3 Filter Bigram using BG_F2 Rules
              df_bigram = df_bigram[df_bigram.bigram.map(lambda x: BG_F2(x,level))]
              
            # D.4 Call only Bigram that meets specified threshold (top 5)
              call_bigram = list(df_bigram["bigram"][0:5])
              freq_callbg = list(df_bigram['freq'][0:5])
              combine_list = list()
              for x in range(0,len(call_bigram)):
                  call1=call_bigram[x]
                  call2=freq_callbg[x] 
                  data = [call1,call2]
                  combine_list.append(data)
              if SocialMediaid=='FB' and level=='Positive':
                  fclust= 7
              elif SocialMediaid=='FB' and level=='Negative':
                  fclust = 5
              sim_ins = round(Sumtable[(Sumtable["NbofK"]==fclust) & (Sumtable["Cluster_ID"]==cluster)]['Mean'].iloc[0],3)
              Bigram_Summary.loc[row_id]=[SocialMediaid,level,cluster,sim_ins,combine_list]
              row_id+=1
    print('Social Media : '+str(SocialMediaid)+' Review_Cat : '+str(level))
 
Bigram_Summary.to_pickle('Bigram_Summary')

Bigram_Summary.to_excel("C:/Users/R/Desktop/Bigram_Summary.xlsx",sheet_name='FB',index = False)

    
   
    