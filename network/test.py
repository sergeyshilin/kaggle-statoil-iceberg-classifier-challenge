import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd

import params
from utils import get_data

from keras.preprocessing.image import ImageDataGenerator

epochs = params.max_epochs
batch_size = params.batch_size
best_weights_path = params.best_weights_path
tta_steps = params.tta_steps

test = pd.read_json('../data/test.json')
X_test, M_test = get_data(test.band_1.values, test.band_2.values, test.inc_angle.values)


def get_data_generator(X, M, batch_size=64):

    while True:
        # suffled indices    
        idx_0 = 0
        # create image generator
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1
        )

        batches = datagen.flow(X, batch_size=batch_size, shuffle=False)

        for batch in batches:
            idx_1 = idx_0 + batch.shape[0]

            yield [ batch, M[idx_0:idx_1] ]

            idx_0 = idx_1
            if idx_1 >= X.shape[0]:
                break


model = params.model_factory(input_shape=X_test.shape[1:])
model.load_weights(filepath=best_weights_path)

predictions = np.zeros((tta_steps, len(X_test)))

test_probas = model.predict([X_test, M_test], batch_size=batch_size, verbose=2)
predictions[0] = test_probas.reshape(test_probas.shape[0])

for i in range(1, tta_steps):
	test_probas = model.predict_generator(
		get_data_generator(X_test, M_test, batch_size=batch_size),
		steps=np.ceil(float(len(X_test)) / float(batch_size)),
		verbose=2
	)
	predictions[i] = test_probas.reshape(test_probas.shape[0])

submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = predictions.mean(axis=0)
submission.to_csv('../submits/submission.csv', index=False)
