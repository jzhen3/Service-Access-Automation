import streamlit as st
import pickle
import pandas as pd
import openpyxl

sheet = openpyxl.load_workbook('dfs_unique.xlsx')['Sheet1']
table = [r[1:] for r in sheet.iter_rows(values_only=True)]
cols = ['team', 'job_Level', 'job_Role', 'app_name']
df = pd.DataFrame(data=table, columns = cols)

def load_model():
    with open('catboost.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

def show_prediction():
    st.title("Service Access Classification")
    st.write("""### Type the employee's information""")

    team = ('Build', 'Run')

    job_Level = ('Intermediate', 'Intern', 'Junior')

    job_Role = ('App Developer',
                'Business Analyst',
                'Data Engineer',
                'Data Scientist',
                'DevOps Engineer',
                'ML Engineer',
                'Product Manager',
                'QA Developer',
                'UX Designer')

    teams = st.selectbox("Team", team)
    job_Levels = st.selectbox("Job Level", job_Level)
    job_Roles = st.selectbox("Job Role", job_Role)

    st.empty()
    st.write("")
    st.empty()

    cols = ['team', 'job_Level', 'job_Role', 'app_name']

    sheet = openpyxl.load_workbook('dfs_unique.xlsx')['Sheet1']
    table = [r[1:] for r in sheet.iter_rows(values_only=True)]
    df = pd.DataFrame(data=table, columns = cols)
    app_name = tuple(df['app_name'])
    app_names = st.selectbox("App Name", app_name)

    b1 = st.button("Identify Access Needs")
    if b1:
        X = pd.DataFrame(data=[[teams, job_Levels, job_Roles, app_names]],
                         columns=cols)
        clf =load_model()
        # return is an array
        label = clf.predict(X)[0]
        status = 'Approved' if label == 1 else 'Denied'
        st.markdown(f"<font size='3'>The service request will be: </font><font size='5'>{status}</font>", unsafe_allow_html=True)

    st.empty()
    st.empty()
    st.empty()
    st.subheader("Please Select App Name If You'd Like to Query")
    b2 = st.button("Query Historic Service Requests")
    sheet_n = openpyxl.load_workbook('query.xlsx')['Sheet1']
    table2 = [r[1:] for r in sheet_n.iter_rows(values_only=True)]
    df2 = pd.DataFrame(data=table2, columns = cols)
    q = df2[(df2['team'] == teams) & (df2['job_Level'] == job_Levels) &
        (df2['job_Role'] ==job_Roles)]['app_name']
    
    if b2:
        st.markdown(f"<font size='3'>Historic Approved Services within specific group:\n\n</font><font size='5'>{list(q)[0]}</font>", unsafe_allow_html=True)