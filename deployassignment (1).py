#!/usr/bin/env python
# coding: utf-8

# In[2]:


import IPython as py


# In[3]:


ipython_instance = py.get_ipython()


# In[7]:


from pickle import dump
from pickle import load
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression


# In[8]:


st.title('Survived Titanic Model')
st.sidebar.header('user Input parameters')


# In[37]:


def user_input_feature():
    Pclass=st.sidebar.selectbox('class',('1','2','3'),key='Pclass')
    Age =st.sidebar.number_input('Insert the Age',key='Age')
    SibSp=st.sidebar.selectbox('SibSp',('1','0'),key='SibSp')
    Parch=st.sidebar.selectbox('Parch',('1','0'),key='Parch')
    Fare=st.sidebar.number_input('Insert the Fare',key='Fare')
    Sex_female=st.sidebar.selectbox('Gender(female)',('1','0'),key='Sex_female')
    Sex_male=st.sidebar.selectbox('Gender (male)',('1','0'),key='Sex_male')
    Embarked_C=st.sidebar.selectbox('Embarked_C',('1','0'),key='Embarked_C')
    Embarked_Q=st.sidebar.selectbox('Embarked_Q',('1','0'),key='Embarked_Q')
    Embarked_S=st.sidebar.selectbox('Embarked_S',('1','0'),key='Embarked_S')
    data={'Pclass':Pclass,'Age':Age,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Sex_female':Sex_female,'Sex_male':Sex_male,
          'Embarked_C':Embarked_C,'Embarked_Q':Embarked_Q,'Embarked_S':Embarked_S}
    features=pd.DataFrame(data,index=[0])
    return features
d=user_input_feature()
st.subheader('user inputs parameters')
st.write(d)


# In[38]:


load_model=load(open("C:/Users/Dayakar Reddy Sura/Downloads/Survived.pkl","rb"))
prediction=load_model.predict(d)
prediction_prob=load_model.predict_proba(d)
st.subheader('predicted results')
st.write('yes' if prediction[0]==0 else 'Not Survived')
st.subheader('prediction probabulity')
st.write(prediction_prob)


# In[ ]:




