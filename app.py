import streamlit as st
from prediction import show_prediction
from viz import show_viz

page =  st.sidebar.selectbox("Select Features", ('Query', 'Assessment'))
show_viz() if page == 'Query' else show_prediction()