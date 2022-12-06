import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#######################################################################################################################
st.title("Heart ❤️ Disease WebApp")
st.write("Welcome to the WebApp of Heart Disease. By providing the following parameters from the patients,\
          this WebApp predicts whether they have Heart Disease or not:")
#######################################################################################################################
age = st.number_input("Patient's age:",
                         min_value=0, max_value=100, value=50, format='%d')
#######################################################################################################################
sex = st.radio(
    "Patient's gender:",
    ('Male', 'Female'), horizontal=True, index=1)
dic = {'Male':1, 'Female':0}
sex = dic[sex]
#######################################################################################################################
cpType = st.selectbox('Does the patient have chest pain?',
                      options=['Typical Angina', 'Atypical Angina',
                               'Non-anginal Pain', 'Asymptomatic'])
dic = {'Typical Angina':0, 'Atypical Angina':1, 'Non-anginal Pain':2, 'Asymptomatic':3}
cpType = dic[cpType]
#######################################################################################################################
rbp = st.slider("Patient's Resting Blood Pressure in mmHg:", 80, 210)
chol = st.slider("Serum Cholestoral in mg/dl:", 100, 600)
#######################################################################################################################
fbs = st.radio("Patient's Fast blood Sugar:",
               ('above 120 mg/dl', 'below 120 mg/dl'), horizontal=True, index=1)
dic = {'above 120 mg/dl':1, 'below 120 mg/dl':0}
fbs = dic[fbs]
#######################################################################################################################
ecg = st.selectbox("Patient's Resting Electrocardiographic Results:",
                   options=['normal', 'having ST-T wave abnormality',
                            "showing probable or definite left ventricular hypertrophy by Estes' criteria"])
dic = {'normal':0, 'having ST-T wave abnormality':1,
       "showing probable or definite left ventricular hypertrophy by Estes' criteria":2}
ecg = dic[ecg]
#######################################################################################################################
mhr = st.slider('maximum heart rate achieved:', 60, 210, 80)
#######################################################################################################################
exang = st.radio(
    "Does the patient have Exercise Induced Angina?",
    ('Yes', 'No'), horizontal=True, index=1)
dic = {'Yes':1, 'No':0}
exang = dic[exang]
#######################################################################################################################
oldpeak = st.slider('ST depression induced by exercise relative to rest', 0.0, 1.0, step=0.01)
#######################################################################################################################
slope = st.radio(
    "The slope of the peak exercise ST segment",
    ('Upsloping', 'Flat', 'Downsloping'), horizontal=True, index=1)
dic = {'Upsloping':1, 'Flat':0, 'Downsloping':-1}
slope = dic[slope]
#######################################################################################################################
nmbv = st.radio(
    "Number of Major Vessels (0-3) Colored by Flourosopy:",
    (0, 1, 2, 3), horizontal=True, index=1)
#######################################################################################################################
thal = st.radio(
    "Thalassemia Disorder",
    ('normal', 'fixed defect', 'reversable defect'), horizontal=True, index=1)
dic = {'normal':0, 'fixed defect':1, 'reversable defect':2}
thal = dic[thal]
#######################################################################################################################
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
#######################################################################################################################
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mms = MinMaxScaler()
ss = StandardScaler()
df[['age', 'rbp', 'chol', 'mhr', 'oldpeak']] = ss.fit_transform(df[['age', 'rbp', 'chol', 'mhr', 'oldpeak']])
df[['cpType', 'ecg', 'slope', 'nmbv', 'thal']] = mms.fit_transform(df[['cpType', 'ecg', 'slope', 'nmbv', 'thal']])
#######################################################################################################################
age, rbp, chol, mhr, oldpeak = ss.transform(pd.DataFrame([[age, rbp, chol, mhr, oldpeak]],
                                                         columns=['age', 'rbp', 'chol', 'mhr', 'oldpeak'])).flatten()
cpType, ecg, slope, nmbv, thal = mms.transform(pd.DataFrame([[cpType, ecg, slope, nmbv, thal]],
                                                            columns=['cpType', 'ecg', 'slope', 'nmbv', 'thal'])).flatten()
inputs = pd.DataFrame([[age, sex, cpType, rbp, chol, fbs, ecg, mhr, exang, oldpeak, slope, nmbv, thal]],
                      columns=['age', 'sex', 'cpType', 'rbp', 'chol', 'fbs', 'ecg', 'mhr',
                               'exang', 'oldpeak', 'slope', 'nmbv', 'thal'])
#######################################################################################################################
y = df["target"]
X = df.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#######################################################################################################################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(inputs)

st.write('\n')
st.markdown('### Result:')
if prediction:
    st.write(" 🟢 It's unlikely that this patient suffers from Heart Disease.")
else:
    st.write(" ⚠️This patient has a high chance of suffering from Heart Disease.")




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
