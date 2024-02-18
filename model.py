team = ['Run', 'Build']
job_Level = ['Intern', 'Junior', 'Intermediate']
job_Role = ['DevOps Engineer', 'App Developer', 'UX Designer', 'Data Engineer', 'ML Engineer', 'Data Scientist', 'QA Developer', 'Product Manager', 'Business Analyst']

from itertools import product
data = []
for comb in list(product(team, job_Level, job_Role)):
  data.append(comb)

import random
random.seed(42)
adjectives = ['Awesome', 'Innovative', 'Fantastic', 'Sleek', 'Dynamic', 'Creative', 'Power', 'Wonder', 'Cool', 'Smart', 'Fun']
nouns = ['App', 'Solution', 'Hub', 'Tool', 'Platform', 'Phase', 'Matrix', 'Space']

app_names = list(product(adjectives, nouns))
app_names = [' '.join(tup) for tup in app_names]
app_names_subset = random.sample(app_names, 30)

import pandas as pd
df = pd.DataFrame(data, columns = ['team', 'job_Level', 'job_Role'])
dfs = pd.concat([df, df, df, df, df, df]).reset_index(drop=True)
rows = len(dfs)
dfs['app_name'] = random.choices(app_names_subset, k = rows)
new_df = dfs.groupby(['team', 'job_Level', 'job_Role'])['app_name'].apply(lambda x: ', '.join(set(x))).reset_index()
dfs_unique= dfs.drop_duplicates().reset_index(drop=True)
dfs_unique['Decision'] = 1

dfs_neg = dfs_unique.copy()
dfs_neg = dfs_neg.groupby(['team', 'job_Level', 'job_Role'])['app_name'].apply(lambda x: list(x)).reset_index()

def list_reduction(a, b):
  return [item for item in a if item not in b ]

dfs_neg['app_name'] = dfs_neg['app_name'].apply(lambda x: list_reduction(app_names_subset, x))
dfs_neg['app_name'] = dfs_neg['app_name'].apply(lambda x: random.sample(x, 5))
dfs_neg = dfs_neg.reset_index(drop=True)

import numpy as np
lol = dfs_neg[['team', 'job_Level', 'job_Role']].values.tolist()
lol = [row for row in lol for _ in range(5)]
app_name_list = [i for a in dfs_neg['app_name'] for i in a]
concat_list = [sub + [add] for sub, add in zip(lol, app_name_list)]
df_dd = pd.DataFrame(concat_list, columns = ['team', 'job_Level', 'job_Role', 'app_name']).drop_duplicates().reset_index(drop=True)
df_dd['Decision'] = 0

complete_table = pd.concat([dfs_unique, df_dd]).reset_index(drop=True)

# split the training and test case
from sklearn.model_selection import train_test_split
features = ['team', 'job_Level', 'job_Role', 'app_name']
X = complete_table[features]
y = complete_table['Decision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=42)

from catboost import CatBoostClassifier
cat_features = list(range(X.shape[1]))
clf = CatBoostClassifier(
    depth = 1,
    iterations=500,
    learning_rate = 0.0001
)
clf.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
)
import pickle
with open("catboost.pkl", 'wb') as f:
   pickle.dump(clf, f)