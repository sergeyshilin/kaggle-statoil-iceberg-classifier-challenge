import sys
from itertools import chain
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from models import EnsembleXGBoostClassifier
from models import EnsembleLightGBMClassifier
from models import EnsembleNNClassifier

np.random.seed = 13

### Parameters of ensembles
num_ensembles = 25
submits_in_ensemble_min = 0.2
submits_in_ensemble_max = 0.8
features_in_ensemble_min = 0.3
features_in_ensemble_max = 0.8
##! Parameters of ensembles



def get_random_model(input_dim):
    models = [
        EnsembleXGBoostClassifier(),
        EnsembleLightGBMClassifier(),
        EnsembleNNClassifier(input_dim)
    ]

    probas = [0.5, 0.5, 0.0]

    return np.random.choice(models, p=probas)


submits = ['0' + str(i) for i in chain(
        range(43, 54),
        range(55, 61),
        range(62, 66),
        range(70, 81),
        range(82, 85)
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
    fraq_submits_in_ensemble = np.random.uniform(low=submits_in_ensemble_min, high=submits_in_ensemble_max)
    fraq_features_in_ensemble = np.random.uniform(low=features_in_ensemble_min, high=features_in_ensemble_max)

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

    ensemble_model = get_random_model(input_dim=X_train.shape[1])

    scaler = StandardScaler()
    ensemble_model.fit(scaler.fit_transform(X_train), y_train)
    result_preds_cv[:, ens_num] = ensemble_model.predict_proba(scaler.transform(X_train))[:, 1]
    result_preds_lb[:, ens_num] = ensemble_model.predict_proba(scaler.transform(X_test))[:, 1]

    print ('ensemble log_loss: {}, classifier: {}'.format(log_loss(y_train, result_preds_cv[:, ens_num]), 
        ensemble_model.get_name()))
    output_string += 'Ensemble classifier: {}\n'.format(ensemble_model.get_name())
    output_string += 'Ensemble log_loss: {}\n'.format(log_loss(y_train, result_preds_cv[:, ens_num]))
    output_string += 'Submits: {}\n'.format([submits[i] for i in submits_subset_idx])
    output_string += 'Features: {}\n'.format(features_subset_cols)
    output_string += 'Ensemble params: {}\n\n'.format(ensemble_model.get_params())


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
