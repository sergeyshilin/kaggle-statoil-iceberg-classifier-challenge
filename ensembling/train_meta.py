import sys
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
import xgboost
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

np.random.seed = 42

submits = ['xgboost_' + str(i) + '.csv' for i in range(10)]#  + ['dense_' + str(i) + '.csv' for i in range(10)]
submits_cv = ['xgboost_cv_' + str(i) + '.csv' for i in range(10)]# + ['dense_cv_' + str(i) + '.csv' for i in range(10)]

dfs_cv = [pd.read_csv(submit_cv) for submit_cv in submits_cv]
dfs = [pd.read_csv(submit) for submit in submits]

features_train = np.zeros((dfs_cv[0].shape[0], len(dfs_cv)))
features_test = np.zeros((dfs[0].shape[0], len(dfs)))
answers_train = dfs_cv[0].is_iceberg.values

for i in range(len(submits)):
    features_train[:, i] = dfs_cv[i].preds.values[:1604]
    features_test[:, i] = dfs[i].is_iceberg.values

# features_train = np.power(np.prod(features_train, axis = 1), 1.0 / features_train.shape[1])
features_train = np.mean(features_train, axis = 1)
print(log_loss(answers_train, features_train))

# preds_test = np.power(np.prod(features_test, axis = 1), 1.0 / features_test.shape[1])
preds_test = np.mean(features_test, axis = 1)
answer_test = pd.DataFrame({'id': dfs[0].id.values, 'is_iceberg': preds_test})
answer_test.to_csv('meta.csv', index = False)
'''
n_xgboosts = int(sys.argv[1])
for xgb in range(n_xgboosts):
    parameters = {'learning_rate' : [np.random.uniform(low = 0.003, high = 0.005)],
                  'max_depth' : [np.random.randint(low = 1, high = 3)],
                  'n_estimators': [np.random.randint(low = 1500, high = 2500)],
                  'min_child_weight':[np.random.randint(low = 1, high = 3)],
                  'gamma':[np.random.uniform(0.3, 0.4)],
                  'subsample':[1], 
                  'colsample_bytree': [np.random.uniform(0.6, 0.9)],
                  'random_state': [np.random.randint(low = 0, high = 30)],
                  'booster': [np.random.choice(['gbtree', 'dart'])], 
                 }
    clf = GridSearchCV(XGBClassifier(), parameters, scoring = 'neg_log_loss', refit = True, cv = 5, n_jobs = -1, verbose = 0)
    clf.fit(features_train, answers_train)
    print(clf.best_params_)
    print(clf.best_score_)

    preds_test = clf.predict_proba(features_test)[:, 1]
    preds_train = clf.predict_proba(features_train)[:, 1]
    answer_test = pd.DataFrame({'id': dfs[0].id.values, 'is_iceberg': preds_test})
    answer_test.to_csv('xgboost_' + str(xgb) + '.csv', index = False)

    answer_train = pd.DataFrame({'preds': preds_train, 'is_iceberg' : dfs_cv[0].is_iceberg.values})
    answer_train.to_csv('xgboost_cv_' + str(xgb) + '.csv', index = False)
'''
