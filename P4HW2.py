import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
df_iris = sns.load_dataset("iris")
import plotly.express as px
st.write("""
# Iris Dataset
How is the target class depends on different dimensions?
""")
fig = px.scatter_3d(df_iris, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
fig.show()