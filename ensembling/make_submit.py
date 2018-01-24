import sys
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

power = float(sys.argv[1])

def transform(preds):
    return preds ** power / (preds ** power + (1.0 - preds) ** power)

with open('submit_id', 'r') as submit_id:
    last_submit_id = int(submit_id.read())

last_submit_id = str(last_submit_id).zfill(3)

ensemble = pd.read_csv('ensembles/ensemble_{}.csv'.format(last_submit_id))
ensemble_cv = pd.read_csv('ensembles_cv/ensemble_cv_{}.csv'.format(last_submit_id))

y_cv = ensemble_cv.is_iceberg
x_cv = ensemble_cv.drop('is_iceberg', axis=1).values.mean(axis=1)

print ('cv log_loss before: {}'.format(log_loss(y_cv, x_cv)))

x_cv_calib = transform(x_cv)
print ('cv log_loss calibration: {}'.format(log_loss(y_cv, x_cv_calib)))

x_cv_clip = np.clip(x_cv, 0.001, 0.999)
print ('cv log_loss clip: {}'.format(log_loss(y_cv, x_cv_clip)))

x_cv_calib_clip = np.clip(transform(x_cv), 0.001, 0.999)
print ('cv log_loss calib+clip: {}'.format(log_loss(y_cv, x_cv_calib_clip)))

submit = pd.read_csv('../data/sample_submission.csv')
submit.is_iceberg = np.clip(transform(ensemble.values.mean(axis=1)), 0.001, 0.999)
submit.to_csv('submits/submission_{}_calib_clip_1_4.csv'.format(last_submit_id), index=False)

