import sys
from itertools import chain
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
import xgboost
import pandas as pd
import numpy as np

np.random.seed = 42

submits = ['0' + str(i) for i in chain(range(43, 54), range(55, 61), range(62, 66), range(70, 76))]

dfs_cv = [pd.read_csv('../submits_cv/submission_cv_' + submit + '.csv') for submit in submits]
dfs = [pd.read_csv('../submits/submission_' + submit + '.csv') for submit in submits]

meta_cv = pd.read_csv('./train_metadata.csv').drop(['id', 'is_iceberg'], axis = 1)
meta = pd.read_csv('./test_metadata.csv').drop(['id', 'is_iceberg'], axis = 1)

meta_features_cv = meta_cv.as_matrix()
meta_features = meta.as_matrix()

features_train = np.zeros((dfs_cv[0].shape[0], len(dfs_cv)))
features_test = np.zeros((dfs[0].shape[0], len(dfs)))
answers_train = dfs_cv[0].is_iceberg.values

for i in range(len(submits)):
    features_train[:, i] = dfs_cv[i].preds.values[:1604]
    features_test[:, i] = dfs[i].is_iceberg.values

features_train = np.concatenate([features_train, meta_features_cv], axis = 1)
features_test = np.concatenate([features_test, meta_features], axis = 1)

n_xgboosts = int(sys.argv[1])
for xgb in range(n_xgboosts):
    features_idxs = np.random.permutation(np.arange(features_train.shape[1])) # [:int(features_train.shape[1] * 0.7)]
    parameters = {'learning_rate' : [np.random.uniform(low = 0.0045, high = 0.005)],
                  'max_depth' : [np.random.randint(low = 1, high = 2)],
                  'n_estimators': [np.random.randint(low = 4000, high = 5000)],
                  'min_child_weight':[np.random.randint(low = 1, high = 2)],
                  'gamma':[np.random.uniform(0.3, 0.4)],
                  'subsample':[1], 
                  'colsample_bytree': [np.random.uniform(0.65, 0.75)],
                  'random_state': [np.random.randint(low = 0, high = 30)],
                  'booster': [np.random.choice(['gbtree'])], 
                 }
    clf = GridSearchCV(XGBClassifier(), parameters, scoring = 'neg_log_loss', refit = True, cv = 5, n_jobs = -1, verbose = 0)
    clf.fit(features_train[:, features_idxs], answers_train[:])
    print(clf.best_params_)
    print(clf.best_score_)

    preds_test = clf.predict_proba(features_test[:, features_idxs])[:, 1]
    preds_train = clf.predict_proba(features_train[:, features_idxs])[:, 1]
    answer_test = pd.DataFrame({'id': dfs[0].id.values, 'is_iceberg': preds_test})
    answer_test.to_csv('xgboost_' + str(xgb) + '.csv', index = False)

    answer_train = pd.DataFrame({'preds': preds_train, 'is_iceberg' : dfs_cv[0].is_iceberg.values})
    answer_train.to_csv('xgboost_cv_' + str(xgb) + '.csv', index = False)
