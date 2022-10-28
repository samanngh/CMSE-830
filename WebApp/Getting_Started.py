import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Heart ❤️ Dataset WebApp")
st.write("Welcome to the WebApp of Heart Dataset! As heart disease is a lack of symptoms problem, \
here, with the help of a dataset, we try to understand which attributes can be the signs of heart disease.")

dataset_info = st.radio(
    "Do you want to know about the source of this dataset?",
    ('Sure', 'No, thanks'), horizontal=True, index=1)
if dataset_info == 'Sure':
    feature_explained = {'cpType': ['Chest Pain Type'], 'rbp': ['Resting Blood Pressure'], \
                         'chol': ['The person\'s cholesterol measurement in mg/dl.'], \
                         'fbs': ['Fast Blood Sugar'], 'ecg': ['Resting electrocardiographic measurement'], \
                         'mhr': ['Maximum Heart Rate Achieved'], 'exang': ['Exercise-induced angina'], \
                         'oldpeak': ['ST depression induced by exercise relative to rest.'], \
                         'slope': ['the slope of the peak exercise ST segment'], \
                         'nmbv': ['The Number of Major Blood Vessels'], \
                         'thal': ['A blood disorder called thalassemia']}
    kir = pd.DataFrame(data=feature_explained)
    st.write('This data set dates from 1988 and consists of four databases: \
    Cleveland, Hungary, Switzerland, and Long Beach V. It contains 303 samples and 14 features. \
    You can see some of these features below.')
    st.dataframe(kir)
else:
    st.write(' ')

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
else:
    st.write(' ')

st.write('The first feature in this dataset is gender. \
Here is the pie chart.')
sns.set_theme()
sns.set_style("white", {"axes.facecolor": "1", 'axes.edgecolor': 'gray'})
sns.set_context("notebook")

cmap = plt.colormaps["Set3"]
outer_colors = cmap([4, 5])
cmap = plt.colormaps["tab20"]
inner_colors = cmap([5, 7, 5, 7])
fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(df['sex'].value_counts(), labels=['Male', 'Female'], colors=outer_colors,
                                   autopct='%.2f%%',
                                   wedgeprops=dict(alpha=0.5, width=0.4), radius=1.5)
autotexts[1]._y -= 0.5
autotexts[0]._y += 0.45
kir = np.array(([df[df['sex'] == 1]['target'].value_counts(), df[df['sex'] == 0]['target'].value_counts()])).flatten()

patches, texts, autotexts = ax.pie(kir, radius=1.5 - 0.4, labels=['No Disease', 'Disease', 'No Disease', 'Disease'],
                                   colors=inner_colors,
                                   autopct='%.2f%%', wedgeprops=dict(width=0.4, alpha=0.5))
texts[0]._x -= 0.35
texts[0]._y -= 0.3
texts[1]._x += 0.47
texts[1]._y += 0.3
texts[2]._x -= 0.25
texts[2]._y += 0.4
texts[3]._x -= 0.5
# texts[3]._y += 0
ax.set(aspect="equal")
st.pyplot(fig)

st.write('')
st.write('By looking at this chart, the first conclusion \
          can be heart disease is probably more common among men. Now, let\'s take a look \
          at another figure. The second figure is the heatmap of correlation matrix.')

mask = np.ones(13) - np.tril(np.ones(13))
f, ax = plt.subplots(figsize=(18, 18))
cmap = sns.color_palette("coolwarm", as_cmap=True)
sns.heatmap(df.corr().iloc[1:, :-1], mask=mask, annot=True, linewidths=.5, fmt='.2f', cmap=cmap, ax=ax,
            center=0, square=True, cbar_kws={"shrink": .5}, vmax=1.0, vmin=-1.0)
st.pyplot(f)

st.write('As you see, `sex`, `cpType`, `mhr`, `exang`, `oldpeak`, `slope`, `nmbv`, and `thal` significantly \
          correlate with heart disease (`target`).')
st.write('')
st.write('Now that you gained some basic information of this dataset, let\'s move to the second \
          and third pages to explore the dataset deeper!')

st.sidebar.markdown("## A WebApp for Heart Disease ⚕️")
st.sidebar.markdown("*[Ehsan Naghavi](naghavis@msu.edu)*")
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
