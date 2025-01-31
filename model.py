import numpy as np
import pandas as pd
teams = ['Analytics', 'Engineering']
job_levels = ['Intern', 'Associate', 'Senior', 'Manager']
job_roles = {
    'Analytics': ['Data Scientist','Product Manager', 'Business Analyst', 'BI Developer'],
    'Engineering': ['DevOps Engineer', 'App Developer', 'UX Designer', 'Data Engineer','QA Developer']
}
tools = {
    'Analytics': ['Tableau', 'Power BI', 'Jira', 'Slack'],
    'Engineering': ['Jenkins', 'GitLab', 'Bitbucket', 'IntelliJ IDEA', 'Docker']
}

num_records = 5000
# Define Sampling Probabilities for Job Levels
job_level_probs = [0.4, 0.3, 0.2, 0.1]
# Randomly assign teams
assigned_teams = np.random.choice(teams, size=num_records)
assigned_job_roles = [np.random.choice(job_roles[team]) for team in assigned_teams]

# Generate the Psuedo dataset
data = {
    'Team': assigned_teams,
    'Job Level': np.random.choice(job_levels, size=num_records, p=job_level_probs),
    'Job Role': [],
    'Tool': [],
    'Approval Status': np.random.choice([0, 1], size=num_records)
}

for team in data['Team']:
    role = np.random.choice(job_roles[team])
    data['Job Role'].append(role)
    tool = np.random.choice(tools[team])
    data['Tool'].append(tool)
# Assign job roles based on team and indicate tools served for users
df = pd.DataFrame(data)


# import numpy as np
# from itertools import product
# data = []
# for comb in list(product(team, job_Level, job_Role)):
#   data.append(comb)

# import random
# random.seed(42)
# adjectives = ['Awesome', 'Innovative', 'Fantastic', 'Sleek', 'Dynamic', 'Creative', 'Power', 'Wonder', 'Cool', 'Smart', 'Fun']
# nouns = ['App', 'Solution', 'Hub', 'Tool', 'Platform', 'Phase', 'Matrix', 'Space']

# app_names = list(product(adjectives, nouns))
# app_names = [' '.join(tup) for tup in app_names]
# app_names_subset = random.sample(app_names, 30)

# import pandas as pd
# df = pd.DataFrame(data, columns = ['team', 'job_Level', 'job_Role'])
# dfs = pd.concat([df, df, df, df, df, df]).reset_index(drop=True)
# rows = len(dfs)
# # simulate the sitaution that each job role have different number of applications
# dfs['app_name'] = random.choices(app_names_subset, k = rows)
# new_df = dfs.groupby(['team', 'job_Level', 'job_Role'])['app_name'].apply(lambda x: ', '.join(set(x))).reset_index()
# dfs_unique= dfs.drop_duplicates().reset_index(drop=True)

# half_length = len(dfs_unique) // 2
# #create a balanced list of 1s and 0s
# balanced_values = [1] * half_length + [0] * (len(dfs_unique) - half_length)
# np.random.shuffle(balanced_values)
# dfs_unique['Decision'] = balanced_values

# complete_table = dfs_unique.reset_index(drop=True)

# split the training and test case
from sklearn.model_selection import train_test_split
features = ['Team', 'Job Level', 'Job Role', 'Tool']
X = df[features]
y = df['Approval Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=42, stratify=y)

from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

cat_model = CatBoostClassifier(
    eval_metric='Recall',
    random_seed=42,
    cat_features=features,
    verbose=0
)

param_grid = {
    'border_count': [140, 150, 160],
    'bagging_temperature': [5.4, 5.5],
    'random_strength': [0.9, 0.92],
    'depth': [5, 6, 7],
    'learning_rate': [0.015, 0.016, 0.017],
    'l2_leaf_reg': [4, 5, 6],
    'iterations': [800, 830]
}

random_search_result = cat_model.randomized_search(
    param_distributions=param_grid,
    X=X_train,
    y=y_train,
    n_iter=100,
    search_by_train_test_split=False,
    refit=True,
    shuffle=True,
    cv=5,
    verbose=True,
    stratified=True
)

print("Best Hyperparameters -> ", random_search_result['params'])
print("Best Test Accuracy Score -> ", cat_model.score(X_test, y_test))

import pickle
with open("catboost.pkl", 'wb') as f:
   pickle.dump(cat_model, f)