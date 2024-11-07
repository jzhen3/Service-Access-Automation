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

    draw_df = df[['app_name', 'team']]
    

    app_team_counts = draw_df.groupby('team')['app_name'].count().reset_index()
    app_team_counts.columns = ['Team_name', 'App Count']

    st.header("Explore the Usage by Different Teams")
    st.write("Bar graph displaying the number of apps by Teams:")

    fig, ax = plt.subplots()

    ax.bar(app_team_counts['Team_name'], app_team_counts['App Count'])
    ax.set_xlabel("App")
    ax.set_ylabel("Counts")
    ax.set_title('Number of Apps by App Name')
    
    st.pyplot(fig)