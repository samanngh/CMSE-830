import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hiplot as hip
import altair as alt
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

df_b = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # Keeping data inside the box

st.sidebar.markdown("## A WebApp for Heart Disease ⚕️")
st.sidebar.markdown("*[Ehsan Naghavi](naghavis@msu.edu)*")
st.sidebar.markdown("CMSE 830, Fall 2022")
st.sidebar.markdown("Michigan State University")

st.markdown("# Exploring the Dataset for further information")
st.write('Up to now, we have gained some basic information about heart disease and \
          how different parameters vary between healthy and unhealthy people. \
          On this page, we try to see what other hidden information could be achieved from this dataset. \
          The outliers are omitted here.')
st.write('')
st.write('Here, you have a combination of the most important features. By the legend on the right, see \
          how they differ between healthy-unhealthy and male-female samples.')

selection = alt.selection_multi(fields=['sex', 'target'])
selection2 = alt.selection_multi(fields=['exang', 'target'])
selection3 = alt.selection_interval(bind='scales')
color = alt.condition(selection,
                      alt.Color('sex:N, target:N', legend=None),
                      alt.value('lightgray'))

scatter1 = alt.Chart(df_b).mark_point().encode(
    alt.Y('slope:N', scale=alt.Scale(zero=False)),
    alt.X('oldpeak:Q', scale=alt.Scale(zero=False)),
    color=color,
    tooltip=[alt.Tooltip('mhr:Q'),
             alt.Tooltip('nmbv:N'),
             alt.Tooltip('chol:Q'),
             alt.Tooltip('fbs:N')
             ]).interactive()
legend1 = alt.Chart(df_b).mark_rect().encode(
    y=alt.Y('target:N', axis=alt.Axis(orient='right')),
    x='sex:N',
    color=color
).add_selection(
    selection, selection3
)
st.altair_chart(scatter1 | legend1, use_container_width=True)

scatter2 = alt.Chart(df_b).mark_point().encode(
    alt.Y('slope:N', scale=alt.Scale(zero=False)),
    alt.X('mhr:Q', scale=alt.Scale(zero=False)),
    color=color,
    tooltip=[alt.Tooltip('nmbv:N'),
             alt.Tooltip('chol:Q'),
             alt.Tooltip('cpType:N'),
             alt.Tooltip('fbs:N')
             ]).interactive()
legend2 = alt.Chart(df_b).mark_rect().encode(
    y=alt.Y('target:N', axis=alt.Axis(orient='right')),
    x='sex:N',
    color=color
).add_selection(
    selection, selection3
)
st.altair_chart(scatter2 | legend2, use_container_width=True)

scatter3 = alt.Chart(df_b).mark_point().encode(
    alt.Y('exang:N', scale=alt.Scale(zero=False)),
    alt.X('mhr:Q', scale=alt.Scale(zero=False)),
    color=color,
    tooltip=[alt.Tooltip('nmbv:N'),
             alt.Tooltip('chol:Q'),
             alt.Tooltip('cpType:N'),
             alt.Tooltip('fbs:N')
             ]).interactive()
legend3 = alt.Chart(df_b).mark_rect().encode(
    y=alt.Y('target:N', axis=alt.Axis(orient='right')),
    x='sex:N',
    color=color
).add_selection(
    selection, selection3
)
st.altair_chart(scatter3 | legend3, use_container_width=True)

st.write('')
st.write('The following plot is also worth to be mentioned. \
          You can see how high the risk of heart disease is when \
          someone has chest pain type zero and Excercise-induced angina.')
barplt = alt.Chart(df_b).mark_bar().encode(
    alt.X('count()'),
    alt.Y('cpType:N'),
    row='exang:N',
    color='target:N')
st.altair_chart(barplt)

st.write('')
st.write('The following plot shows that heart disease decreases the `mhr` in younger ages.')
scatter4 = alt.Chart(df_b).mark_point().encode(
    alt.X('age:Q', scale=alt.Scale(zero=False)),
    alt.Y('mhr:Q', scale=alt.Scale(zero=False)),
    color='target:N',
    tooltip=[alt.Tooltip('mhr:Q'),
             alt.Tooltip('target:N'),
             alt.Tooltip('cpType:N'),
             alt.Tooltip('sex:N')
             ])
fig4 = scatter4 + scatter4.transform_regression('age', 'mhr', method='quad', groupby=['target']).mark_line()
st.altair_chart(fig4, use_container_width=True)

st.write('')
st.write('And finally, the parallel plot! As our data\'s dimension is relatively high, \
          the parallel plot can help us to learn more from the data. If you find something new from \
          this dataset, please share it with me!')
prl_plot = hip.Experiment.from_dataframe(df_b).to_streamlit(ret="selected_uids", key="hip").display()

st.write('')
st.write('Thanks for using my WebApp. Please feel free to share your thought and suggestions with me.')
