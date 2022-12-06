import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Information on the Dataset used for this WebApp")
st.write("On this page, you get familiar with the dataset used for training the predictive model.\
          This data set dates from 1988 and consists of four databases: \
          Cleveland, Hungary, Switzerland, and Long Beach V. It contains 303 samples and 14 features.\
          Some of the features are explained below:")

feature_explained = {'cpType': ['Chest Pain Type'], 'rbp': ['Resting Blood Pressure'], \
                     'chol': ['The person\'s cholesterol measurement in mg/dl.'], \
                     'fbs': ['Fast Blood Sugar'], 'ecg': ['Resting electrocardiographic measurement'], \
                     'mhr': ['Maximum Heart Rate Achieved'], 'exang': ['Exercise-induced angina'], \
                     'oldpeak': ['ST depression induced by exercise relative to rest.'], \
                     'slope': ['the slope of the peak exercise ST segment'], \
                     'nmbv': ['The Number of Major Blood Vessels'], \
                     'thal': ['A blood disorder called thalassemia']}

kir = pd.DataFrame(data=feature_explained)
st.dataframe(kir)

df = pd.read_csv('heart.csv')
df = df.rename({'cp': 'cpType', 'trestbps': 'rbp', 'restecg': 'ecg', 'thalach': 'mhr', 'ca': 'nmbv'}, axis='columns')
df['slope'] = df['slope'].replace(0, -1)
df['slope'] = df['slope'].replace(1, 0)
df['slope'] = df['slope'].replace(2, 1)
df['thal'] = df['thal'].replace(1, -1)
df['thal'] = df['thal'].replace(0, None)
df['thal'] = df['thal'].replace(2, 0)
df['thal'] = df['thal'].replace(3, 1)
df['thal'] = df['thal'].replace(-1, 2)
df['ecg'] = df['ecg'].replace(0, -1)
df['ecg'] = df['ecg'].replace(1, 0)
df['ecg'] = df['ecg'].replace(2, 1)
df['ecg'] = df['ecg'].replace(-1, 2)
df = df.dropna()
df = df.drop_duplicates()

dataset_view = st.radio(
    "Do you want to see the dataset?",
    ('Sure', 'No, thanks'), horizontal=True, index=1)

if dataset_view == 'Sure':
    st.dataframe(df)
    value_explanation = st.radio(
        "Do you want to know what these values mean?",
        ('Sure', 'No, thanks'), horizontal=True, index=1)
    if value_explanation == 'Sure':
        st.write('#### Explaining the Values of the Features:')
        st.markdown("""
            * Categorical Features:
                * Sex: Male=1, Female=0
                * cpType: ASY=3, NAP=2, ATA=1, TA=0
                * fbs: ( fbs > 120 mg/dl)=1, otherwise=0
                * ecg: Normal=0, LVH=1, ST=2
                * exang: Angina=1, No Angina=0
                * slope: Up=1, Flat=0, Down=-1
                * nmbv
                * thal: normal=0, fixed defect=1, reversable defect=2
            * Numerical Features:
                * Age
                * rbp
                * Chol
                * mhr
                * oldpeak
                """)
    st.write(' ')

else:
    st.write(' ')

st.sidebar.markdown("## A WebApp for Heart Disease ⚕️")
st.sidebar.markdown('<a href="mailto:naghavis@msu.edu"> *Ehsan Naghavi* </a>', unsafe_allow_html=True)
st.sidebar.markdown("CMSE 830, Fall 2022")
st.sidebar.markdown("Michigan State University")
rm = ['Cardiovascular diseases (CVDs) are the leading cause of death globally.',
      'An estimated 17.9 million people died from CVDs in 2019, representing 32% \
      of all global deaths. Of these deaths, 85% were due to heart attack and stroke.',
      'Over three quarters of CVD deaths take place in low- and middle-income countries.',
      'Out of the 17 million premature deaths (under the age of 70) due to noncommunicable diseases in 2019, \
      38% were caused by CVDs.',
      'Most cardiovascular diseases can be prevented by addressing behavioural risk factors \
      such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol.',
      'It is important to detect cardiovascular disease as early as possible so that management \
      with counselling and medicines can begin.']

st.sidebar.write('')
st.sidebar.write('🔴', rm[np.random.randint(len(rm))])
