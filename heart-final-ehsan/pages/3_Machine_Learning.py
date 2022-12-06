import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hiplot as hip
import altair as alt
import pandas as pd
from scipy import stats
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
y = df["target"]
X = df.drop(['target'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#######################################################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from mlxtend.classifier import StackingCVClassifier
#######################################################################################################################
GNB = GaussianNB()
RF = RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3)
svm = SVC(gamma=0.1, C=1)
KNN = KNeighborsClassifier(7)
#######################################################################################################################
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
st.title('Classification Details')
st.write('This page explains how the predictive model on the first page got trained. \
          These following steps were taken:')
st.markdown("""
            1. The categorical features got normalized and the continuous features got standardized.
            2. The dataset got separated to two subsets (Training Dataset and Test Dataset).
            3. Different classifiers were trained and the hyperparameters were adjust to get the best results.
            4. Finally, the ensembling technique was used to improve the accuracy of the model. 
                """)
st.write("You can see the performance of each classifier below:")
#######################################################################################################################
clf_names = ['Logistic Regression', 'K-Nearest Neighbors', 'SVM', 'Gaussian Process Classifier',
             'Random Forest', 'Decision Tree', 'MLP Classifier', 'Gaussian NB',
             'Quadratic Discriminant Analysis', 'Stacking CV Classifier']

dic = {'Logistic Regression':LogisticRegression(random_state=0),
       'K-Nearest Neighbors':KNeighborsClassifier(7),
       'SVM':SVC(gamma=0.1, C=1),
       'Gaussian Process Classifier':GaussianProcessClassifier(),
       'Random Forest':RandomForestClassifier(max_depth=10, n_estimators=10, max_features=3),
       'Decision Tree':DecisionTreeClassifier(max_depth=3),
       'MLP Classifier':MLPClassifier(alpha=0.01, max_iter=10000),
       'Gaussian NB':GaussianNB(),
       'Quadratic Discriminant Analysis':QuadraticDiscriminantAnalysis(),
       'Stacking CV Classifier':StackingCVClassifier(classifiers=[KNN,svm,GNB],meta_classifier= RF,random_state=0)}

clf = st.selectbox('Select the classifier you want to see its performance:',
                   options=clf_names)
clf = dic[clf]
clf.fit(X_train,y_train)
predicted = clf.predict(X_test)
st.write('accuracy score = ', accuracy_score(y_test, predicted))
confmat = confusion_matrix(y_test, predicted)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                confmat.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     confmat.flatten()/np.sum(confmat)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
fig, ax = plt.subplots()
ax = sns.heatmap(confmat, annot=labels, fmt='', cmap='Blues')
st.pyplot(fig)
report = classification_report(y_test, predicted, output_dict=True)
df = pd.DataFrame(report).transpose()
st.dataframe(df)
st.write('')
st.write('In the final model, *Stacking CV Classifier* which showed the best performance, has been employed.')
#######################################################################################################################
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

st.sidebar.markdown("## A WebApp for Heart Disease ⚕️")
st.sidebar.markdown('<a href="mailto:naghavis@msu.edu"> *Ehsan Naghavi* </a>', unsafe_allow_html=True)
st.sidebar.markdown("CMSE 830, Fall 2022")
st.sidebar.markdown("Michigan State University")
st.sidebar.write('')
st.sidebar.write('🔴', rm[np.random.randint(len(rm))])