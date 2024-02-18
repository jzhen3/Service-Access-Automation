import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

def show_viz():
    st.header("Explore the whole query")
    sheet = openpyxl.load_workbook('dfs_unique.xlsx')['Sheet1']
    table = [r[1:] for r in sheet.iter_rows(values_only=True)]
    cols = ['team', 'job_Level', 'job_Role', 'app_name']
    df = pd.DataFrame(data=table, columns = cols)

    sheet_n = openpyxl.load_workbook('query.xlsx')['Sheet1']
    table2 = [r[1:] for r in sheet_n.iter_rows(values_only=True)]
    df2 = pd.DataFrame(data=table2, columns = cols)

    d2 = st.dataframe(df2.iloc[1:])

    st.header("Explore the Usage by Different Apps")
    fig, ax = plt.subplots()
    draw_df = df[['app_name', 'team']]
    
    apps = [i for i in df['app_name']]
    app_count = draw_df.groupby('app_name')['team'].count().to_list()

    # draw_df['app_count'] = draw_df.groupby('app_name')['team'].count()

    plt.barh(y = apps, x= app_count, title='App Usage', width = 0.4)
    
    plt.show()
