import sys
from itertools import chain
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

np.random.seed = 42

### Parameters of ensembles
num_ensembles = 25
fraq_submits_in_ensemble = 0.5
fraq_features_in_ensemble = 0.6
##! Parameters of ensembles

## Define universal ensemble classifier (interface) and 
## all other classifiers we might use
class EnsembleClassifier:
    def __init__(self, classifier, parameters):
        self.parameters = parameters
        self.classifier = classifier(**parameters)

    def fit(self, X, y):
        return self.classifier.fit(X_train, y_train)

    def predict(self, data):
        return self.classifier.predict(data)

    def predict_proba(self, data):
        return self.classifier.predict_proba(data)

    def get_params(self):
        return self.parameters


class EnsembleXGBoostClassifier(EnsembleClassifier):
    def __init__(self):
        parameters = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'learning_rate': np.random.uniform(low=0.0045, high=0.005),
            'max_depth': np.random.randint(low=1, high=3),
            'n_estimators': np.random.randint(low = 4000, high = 5000), ## equals to num_rounds in xgb
            'min_child_weight': np.random.randint(low=1, high=3),
            'gamma': np.random.uniform(0.0, 0.2),
            'subsample': np.random.uniform(0.8, 1.0),
            'colsample_bytree': np.random.uniform(0.7, 0.9),
            'random_state': np.random.randint(low=0, high=30)
         }

        EnsembleClassifier.__init__(self, xgb.XGBClassifier, parameters)

    def fit(self, X, y):
        return self.classifier.fit(X, y, eval_metric='logloss', verbose=True)


def get_random_model():
    models = [
        EnsembleXGBoostClassifier()
    ]

    return np.random.choice(models)


submits = ['0' + str(i) for i in chain(
        range(43, 54),
        range(55, 61),
        range(62, 66),
        range(70, 80)
    )
]

submits_cv = [pd.read_csv('../submits_cv/submission_cv_' + submit + '.csv') for submit in submits]
submits_lb = [pd.read_csv('../submits/submission_' + submit + '.csv') for submit in submits]

meta_cv = pd.read_csv('./train_metadata.csv').drop(['id', 'is_iceberg'], axis = 1)
meta_lb = pd.read_csv('./test_metadata.csv').drop(['id', 'is_iceberg'], axis = 1)

result_preds_cv = np.zeros((1604, num_ensembles))
result_preds_lb = np.zeros((len(meta_lb), num_ensembles))
output_string = ""

y_train = submits_cv[0][:1604].is_iceberg

for ens_num in range(num_ensembles):
    submits_subset_size = int(fraq_submits_in_ensemble * len(submits_cv))
    features_subset_size = int(fraq_features_in_ensemble * len(meta_cv.columns))
    submits_subset_idx = np.random.choice(range(len(submits_cv)), size=submits_subset_size, replace=False)
    features_subset_cols = np.random.choice(meta_cv.columns, size=features_subset_size, replace=False)

    submits_subset_cv = [submits_cv[i][:1604].preds.rename('preds_{}'.format(submits[i])) for i in submits_subset_idx]
    submits_subset_lb = [submits_lb[i].is_iceberg.rename('preds_{}'.format(submits[i])) for i in submits_subset_idx]
    features_subset_cv = meta_cv[features_subset_cols]
    features_subset_lb = meta_lb[features_subset_cols]

    X_train = pd.concat([*submits_subset_cv, features_subset_cv], axis=1)
    X_test = pd.concat([*submits_subset_lb, features_subset_lb], axis=1)

    ensemble_model = get_random_model()

    ensemble_model.fit(X_train, y_train)
    result_preds_cv[:, ens_num] = ensemble_model.predict_proba(X_train)[:, 1]
    result_preds_lb[:, ens_num] = ensemble_model.predict_proba(X_test)[:, 1]

    print ('ensemble log_loss: {}'.format(log_loss(y_train, result_preds_cv[:, ens_num])))
    output_string += 'ensemble log_loss: {}\n'.format(log_loss(y_train, result_preds_cv[:, ens_num]))
    output_string += 'submits: {}\n'.format([submits[i] for i in submits_subset_idx])
    output_string += 'features: {}\n'.format(features_subset_cols)
    output_string += 'XGB params: {}\n\n'.format(ensemble_model.get_params())


## ========================= MAKE CV AND LB SUBMITS ========================= ##
with open('submit_id', 'r') as submit_id:
    last_submit_id = int(submit_id.read())

last_submit_id += 1

with open('ensembles_cv/ensemble_cv_{0:0>3}.info'.format(last_submit_id), 'w') as info_output:
    info_output.write(output_string)

column_names = ['ensemble_{}'.format(ens_id + 1) for ens_id in range(num_ensembles)]

submission_cv = pd.DataFrame(result_preds_cv, columns=column_names)
submission_cv['is_iceberg'] = y_train
submission_cv.to_csv('ensembles_cv/ensemble_cv_{0:0>3}.csv'.format(last_submit_id), index=False)

submission = pd.DataFrame(result_preds_lb, columns=column_names)
submission.to_csv('ensembles/ensemble_{0:0>3}.csv'.format(last_submit_id), index=False)

with open('submit_id', 'w') as submit_id:
    submit_id.write(str(last_submit_id))
