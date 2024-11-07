team = ['Analytics', 'Engineering']
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


# dfs_unique['Decision'] = 1

# dfs_neg = dfs_unique.copy()
# dfs_neg = dfs_neg.groupby(['team', 'job_Level', 'job_Role'])['app_name'].apply(lambda x: list(x)).reset_index()

# def list_reduction(a, b):
#   return [item for item in a if item not in b ]

# dfs_neg['app_name'] = dfs_neg['app_name'].apply(lambda x: list_reduction(app_names_subset, x))
# dfs_neg['app_name'] = dfs_neg['app_name'].apply(lambda x: random.sample(x, 5))
# dfs_neg = dfs_neg.reset_index(drop=True)

# import numpy as np
# lol = dfs_neg[['team', 'job_Level', 'job_Role']].values.tolist()
# lol = [row for row in lol for _ in range(5)]
# app_name_list = [i for a in dfs_neg['app_name'] for i in a]
# concat_list = [sub + [add] for sub, add in zip(lol, app_name_list)]
# df_dd = pd.DataFrame(concat_list, columns = ['team', 'job_Level', 'job_Role', 'app_name']).drop_duplicates().reset_index(drop=True)
# df_dd['Decision'] = 0

half_length = len(dfs_unique) // 2
#create a balanced list of 1s and 0s
balanced_values = [1] * half_length + [0] * (len(dfs_unique) - half_length)
import numpy as np
np.random.shuffle(balanced_values)
dfs_unique['Decision'] = balanced_values

# complete_table = pd.concat([dfs_unique, df_dd]).reset_index(drop=True)
complete_table = dfs_unique.reset_index(drop=True)

# split the training and test case
from sklearn.model_selection import train_test_split
features = ['team', 'job_Level', 'job_Role', 'app_name']
X = complete_table[features]
y = complete_table['Decision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=42)

from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
# cat_features = list(range(X.shape[1]))

cat_model = CatBoostClassifier(
    cat_features=features,
    verbose=0
)

param_grid = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 1, 2, 5],
    'random_strength': [0.5, 1, 2],
}

random_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='f1',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# clf = CatBoostClassifier(
#     depth = 1,
#     iterations=500,
#     learning_rate = 0.0001
# )
# clf.fit(
#     X_train, y_train,
#     cat_features=cat_features,
#     eval_set=(X_test, y_test),
# )

print("Best Hyperparameters -> ", random_search.best_params_)
print("Best Score -> ", random_search.best_score_)

best_cat_model = random_search.best_estimator_
best_cat_model.fit(X_train, y_train)

import pickle
with open("catboost.pkl", 'wb') as f:
   pickle.dump(best_cat_model, f)