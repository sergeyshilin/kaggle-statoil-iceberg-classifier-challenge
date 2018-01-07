from itertools import chain
import sys
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
import xgboost
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

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

num_nets = int(sys.argv[1])

for net in range(num_nets):
    batch_size = 128
    num_classes = 2
    epochs = 120

    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim = features_train.shape[1]))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss = keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(lr = 0.00001),
                  metrics = ['accuracy'])
    
    idxs = np.arange(1604)
    np.random.shuffle(idxs)

    model.fit(features_train[idxs[:1200]], answers_train[idxs[:1200]],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(features_train[idxs[1200:]], answers_train[idxs[1200:]]))

    score = model.evaluate(features_train[idxs[1200:]], answers_train[idxs[1200:]], verbose=0)
    print('Test loss:', score[0])
    
    preds_test = model.predict(features_test)[:, 0]
    preds_train = model.predict(features_train)[:, 0]
    answer_test = pd.DataFrame({'id': dfs[0].id.values, 'is_iceberg': preds_test})
    answer_test.to_csv('dense_' + str(net) + '.csv', index = False)

    answer_train = pd.DataFrame({'preds': preds_train, 'is_iceberg' : dfs_cv[0].is_iceberg.values})
    answer_train.to_csv('dense_cv_' + str(net) + '.csv', index = False)
