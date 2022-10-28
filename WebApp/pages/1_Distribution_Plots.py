import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
st.sidebar.markdown("## A WebApp for Heart Disease ⚕️")
st.sidebar.markdown("*[Ehsan Naghavi](naghavis@msu.edu)*")
st.sidebar.markdown("CMSE 830, Fall 2022")
st.sidebar.markdown("Michigan State University")

st.markdown("# What can predict heart disease❓")
st.write('On this page, we use distribution plots of different features to see how they vary \
          between people with and without heart disease. For categorical variables, you\'ll have \
          the bar plots. The violin plots will be employed to show you the distribution of \
          continuous quantities.')

n_plot = st.number_input('First, select how many distribution plots you want to see?',
                         min_value=0, max_value=12, format='%d')
st.write('')

age, g1 = plt.subplots()
g1 = sns.violinplot(data=df, x="sex", y="age", hue="target", palette="husl")
for violin in g1.collections[::2]:
    violin.set_alpha(0.7)
g1.axes.legend()
leg1 = g1.get_legend()
for t, l in zip(leg1.texts, ['Disease', 'No Disease']):
    t.set_text(l)
g1.axes.set_xticklabels(['Female', 'Male'])
g1.axes.set_xlabel('Gender')
sns.despine()

cpType = sns.displot(df.astype(str), x="cpType", col="sex", hue="target", multiple="stack",
                     legend=False, palette="husl", alpha=0.5, aspect=0.75)
axes1 = cpType.axes.flatten()
axes1[0].set_title("Male")
axes1[1].set_title("Female")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(cpType)

rbp, g1 = plt.subplots()
g1 = sns.violinplot(data=df, x="sex", y="rbp", hue="target", palette="husl")
for violin in g1.collections[::2]:
    violin.set_alpha(0.7)
g1.axes.legend()
leg1 = g1.get_legend()
for t, l in zip(leg1.texts, ['Disease', 'No Disease']):
    t.set_text(l)
g1.axes.set_xticklabels(['Female', 'Male'])
g1.axes.set_xlabel('Gender')
sns.despine()
# st.pyplot(rbp)

chol, g2 = plt.subplots()
g2 = sns.violinplot(data=df, x="sex", y="chol", hue="target", palette="husl", alpha=0.4)
for violin in g2.collections[::2]:
    violin.set_alpha(0.7)
handles, labels = g2.axes.get_legend_handles_labels()
g2.axes.legend()
leg2 = g2.get_legend()
for t, l in zip(leg2.texts, ['Disease', 'No Disease']):
    t.set_text(l)
g2.axes.set_xticklabels(['Female', 'Male'])
g2.axes.set_xlabel('Gender')
sns.despine()
# st.pyplot(chol)

fbs = sns.displot(df.astype(str), x="fbs", col="sex", hue="target", multiple="stack",
                  legend=False, palette="husl", alpha=0.5, aspect=0.5)
axes2 = fbs.axes.flatten()
axes2[0].set_title("Male")
axes2[1].set_title("Female")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(fbs)

ecg = sns.displot(df, x="ecg", col="sex", hue="target", multiple="stack", kind='hist', discrete=True,
                  legend=False, palette="husl", alpha=0.5, aspect=0.5)
axes3 = ecg.axes.flatten()
axes3[0].set_title("Female")
axes3[1].set_title("Male")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(ecg)

mhr, g3 = plt.subplots()
g3 = sns.violinplot(data=df, x="sex", y="mhr", hue="target", palette="husl", alpha=0.4)
for violin in g3.collections[::2]:
    violin.set_alpha(0.7)
g3.axes.legend()
leg3 = g3.get_legend()
for t, l in zip(leg3.texts, ['Disease', 'No Disease']):
    t.set_text(l)
g3.axes.set_xticklabels(['Female', 'Male'])
g3.axes.set_xlabel('Gender')
# st.pyplot(mhr)

exang = sns.displot(df.astype(str), x="exang", col="sex", hue="target", multiple="stack",
                    legend=False, palette="husl", alpha=0.5, aspect=0.5)
axes4 = exang.axes.flatten()
axes4[0].set_title("Male")
axes4[1].set_title("Female")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(exang)

oldpeak, g4 = plt.subplots()
g4 = sns.violinplot(data=df, x="sex", y="oldpeak", hue="target", palette="husl", alpha=0.4)
for violin in g4.collections[::2]:
    violin.set_alpha(0.7)
g4.axes.legend()
leg4 = g4.get_legend()
for t, l in zip(leg4.texts, ['Disease', 'No Disease']):
    t.set_text(l)
g4.axes.set_xticklabels(['Female', 'Male'])
g4.axes.set_xlabel('Gender')
sns.despine()
# st.pyplot(oldpeak)

nmbv = sns.displot(df, x="nmbv", col="sex", hue="target", multiple="stack", kind='hist', discrete=True,
                   legend=False, palette="husl", alpha=0.5, aspect=0.75)
axes5 = nmbv.axes.flatten()
axes5[0].set_title("Female")
axes5[1].set_title("Male")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(nmbv)

slope = sns.displot(df, x="slope", col="sex", hue="target", multiple="stack", kind='hist', discrete=True,
                    legend=False, palette="husl", alpha=0.5, aspect=0.5)
axes6 = slope.axes.flatten()
axes6[0].set_title("Female")
axes6[1].set_title("Male")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(slope)

thal = sns.displot(df, x="thal", col="sex", hue="target", multiple="stack", kind='hist', discrete=True,
                   legend=False, palette="husl", alpha=0.5, aspect=0.5)
axes7 = thal.axes.flatten()
axes7[0].set_title("Female")
axes7[1].set_title("Male")
plt.legend(labels=['No Disease', 'Disease'])
# st.pyplot(thal)

string1 = 'As expected, getting older increases the chance of getting heart disease.'
string2 = 'The chest pain type zero can be very serious while the rest three are not typically important.'
string3 = 'People with higher blood pressure have a larger risk of getting heart disease.'
string4 = 'In contrast to our expectations, Cholesterol does not vary significantly \
           between healthy and unhealthy people.'
string5 = 'The second surprise is that the fast blood sugar does not affect on the risk of the heart disease, \
           at least based on these observations.'
string6 = 'Resting electrocardiographic measurement is almost the same between the two groups. \
           Therefore, it cannot be a helpful parameter for heart disease diagnosis.'
string7 = 'People with heart disease have a lower maximum heart rate achieved as expected.'
string8 = 'Exercise-induced angina could be considered as a heart disease sign.'
string9 = 'Oldpeak differs between healthy and unhealthy cases.'
string11 = 'With heart disease, the number of major blood vessels increases.'
string10 = 'If the slope of the peak exercise ST segment is not positive, \
            then the physicians should be more careful in their other examinations.'
string12 = 'Thalassemia is a red flag for heart disease.'

feature_explained = {'age': string1, 'cpType': string2, \
                     'rbp': string3, 'chol': string4, \
                     'fbs': string5, 'ecg': string6, \
                     'mhr': string7, 'exang': string8, \
                     'oldpeak': string9, 'slope': string10, \
                     'nmbv': string11, 'thal': string12}
plot_list = []

for i in range(n_plot):
    st.write('Now select feature ', i + 1, ' :')
    feature_name = st.selectbox('Select the feature:',
                                options=df.columns.drop(['sex', 'target']).drop(plot_list), label_visibility='hidden')
    st.write(feature_name)
    st.pyplot(locals()[feature_name])
    st.write(feature_explained[feature_name])
    plot_list.append(feature_name)

st.write('So, did you find the information you were looking for? If not, no worries.\
          There is still one more page to explore!')
