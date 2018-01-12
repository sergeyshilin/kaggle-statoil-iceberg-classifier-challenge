import numpy as np
np.random.seed = 42

import xgboost as xgb
from lightgbm import LGBMClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier


## Define universal ensemble classifier (interface) and
## all other classifiers we might use
class EnsembleClassifier:
    def __init__(self, classifier=None):
        if classifier:
            self.classifier = classifier(**self.parameters)

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, data):
        return self.classifier.predict(data)

    def predict_proba(self, data):
        return self.classifier.predict_proba(data)

    def get_params(self):
        return self.parameters

    def get_name(self):
        return self.name


class EnsembleXGBoostClassifier(EnsembleClassifier):
    def __init__(self, input_dim=None):
        self.name = 'XGBoost'
        self.parameters = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'learning_rate': np.random.uniform(low=0.0045, high=0.005),
            'max_depth': np.random.randint(low=1, high=3),
            'n_estimators': np.random.randint(low=4000, high=8000), ## equals to num_rounds in xgb
            'min_child_weight': np.random.randint(low=1, high=3),
            'gamma': np.random.uniform(0.0, 0.2),
            'subsample': np.random.uniform(0.8, 1.0),
            'colsample_bytree': np.random.uniform(0.7, 1.0),
            'random_state': np.random.randint(low=0, high=30)
         }

        EnsembleClassifier.__init__(self, xgb.XGBClassifier)

    def fit(self, X, y):
        return self.classifier.fit(X, y, eval_metric='logloss', verbose=True)


class EnsembleLightGBMClassifier(EnsembleClassifier):
    def __init__(self, input_dim=None):
        self.name = "LightGBM"
        self.parameters = {
            'boosting_type': np.random.choice(['gbdt', 'dart']),
            'objective': 'binary',
            'learning_rate' : np.random.uniform(low=0.045, high=0.05),
            'max_depth' : np.random.randint(low=1, high=2),
            'n_estimators': np.random.randint(low=2000, high=3500),
            'subsample': np.random.uniform(0.8, 1.0),
            'colsample_bytree': np.random.uniform(0.7, 1.0),
            'random_state': np.random.randint(low=0, high=30),
        }

        EnsembleClassifier.__init__(self, LGBMClassifier)

    def fit(self, X, y):
        return self.classifier.fit(X, y, eval_metric='logloss', verbose=True)


class EnsembleNNClassifier(EnsembleClassifier):
    def __init__(self, input_dim=None):
        self.input_dim = input_dim
        self.name = "Keras NN"
        self.parameters = {
            'learning_rate': np.random.uniform(low=0.045, high=0.05),
            'epochs': np.random.randint(low=150, high=180),
            'neurons_1': np.random.randint(low=34, high=45),
            'neurons_2': np.random.randint(low=10, high=15),
            'dropout_1': np.random.uniform(0.01, 0.1),
            'dropout_2': np.random.uniform(0.15, 0.20),
            'initializer': 'glorot_uniform'
        }

        EnsembleClassifier.__init__(self)

        self.classifier = KerasClassifier(build_fn=self._create_model, epochs=self.parameters['epochs'],
            batch_size=64, verbose=0)

    def _create_model(self):
        # create model
        optimizer = Adam(lr=self.parameters['learning_rate'])
        model = Sequential()
        model.add(Dense(self.parameters['neurons_1'], input_dim=self.input_dim, 
            kernel_initializer=self.parameters['initializer'], activation='relu'))
        model.add(Dropout(self.parameters['dropout_1']))
        model.add(Dense(self.parameters['neurons_2'], 
            kernel_initializer=self.parameters['initializer'], activation='relu'))
        model.add(Dropout(self.parameters['dropout_2']))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model
