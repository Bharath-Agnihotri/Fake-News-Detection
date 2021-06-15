#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessory packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


#reading the data
df=pd.read_csv(r'C:\Users\admin\Desktop\Project Data\news.csv')
df.shape
print(df.head())


# In[3]:


#DataFlair get the labels
labels=df.label
print(labels.head())


# In[4]:


#DataFlair split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=7)


# In[5]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)


# In[6]:


#fit and transform
train=tfidf_vectorizer.fit_transform(x_train)
test=tfidf_vectorizer.transform(x_test)


# In[8]:


#Initialize PassiveAgressiveClassifier
pas_ag_cl=PassiveAggressiveClassifier(max_iter=50)
pas_ag_cl.fit(train,y_train)


# In[9]:


y_pred=pas_ag_cl.predict(test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy:{round(score*100)}%')


# In[10]:


#Confusion_Matrix
confusion_matrix(y_test,y_pred)


# In[ ]:





# In[ ]:




