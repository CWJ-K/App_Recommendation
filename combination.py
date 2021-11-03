"""
Created on Thursday  Feb 14, 2019
@title : Media and Web Analytics
@Group : 2 - (Chen Wan Ju, Sweetie Anang, Gandes Aisyaharum, Nessya Callista, Mirza Miftanula)
@author: Chen Wan Ju
@Obj   : Combine all data from customers' review and label ther source and the app of data(e.g. ios, Android, Argos, Amazons) 
"""
# Name of cloumns: App Content_Review Date_Rev ID Reviewer Star_Rating Title platform


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
 
data = ['Amazon_Android','Amazon_ios','Argos_Android','Argos_ios'] 
database = pd.DataFrame()


for i in range(len(data)):
        df = pd.read_excel(f'{data[i]}.xlsx')
        if 'ios' in data[i]:
                df['platform'] = 'ios'
                if 'Amazon' in data[i]:
                        df['App'] = 'Amazon'
                if 'Argos' in data[i]:
                        df['App'] = 'Argos'
        if 'Andro' in data [i]:
                df['platform'] = 'Android'
                if 'Amazon' in data[i]:
                        df['App'] = 'Amazon'
                if 'Argos' in data[i]:
                        df['App'] = 'Argos'
        vars()[f'{data[i]}']=df

data1 = [Amazon_Android,Amazon_ios,Argos_Android,Argos_ios]         
database = pd.concat(data1)

database.to_excel('C:/Users/Regem/Desktop/Customers_reviews.xlsx')






