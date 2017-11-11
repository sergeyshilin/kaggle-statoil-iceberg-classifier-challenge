import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd

import params
from utils import get_data

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

epochs = params.max_epochs
batch_size = params.batch_size
best_weights_path = params.best_weights_path

test = pd.read_json('data/test.json')
X_test = get_data(test.band_1.values, test.band_2.values, test.inc_angle.values)

model = params.model_factory(input_shape=X_test.shape[1:])
model.load_weights(filepath=best_weights_path)

predicted_test = model.predict_proba(X_test)

submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('submits/submission.csv', index=False)
