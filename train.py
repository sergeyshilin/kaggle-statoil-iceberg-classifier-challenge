import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd

import params
from utils import get_data

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

epochs = params.max_epochs
batch_size = params.batch_size
model = params.model_factory()
validation_split = params.validation_split
best_weights_path = params.best_weights_path

train = pd.read_json('data/train.json')
# train = train[train.inc_angle != "na"]
X_train = get_data(train.band_1.values, train.band_2.values)
y_train = train['is_iceberg']

xtr, xcv, ytr, ycv = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        min_delta=1e-4,
        mode='min'
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
        verbose=1,
        epsilon=1e-4,
        mode='min'
    ),

    ModelCheckpoint(
        monitor='val_loss',
        filepath=best_weights_path,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
]

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    channel_shift_range=0.1,
    zoom_range=0.1
)

model.summary()

model.fit_generator(
    datagen.flow(xtr, ytr, batch_size=batch_size),
    steps_per_epoch=np.ceil(float(len(X_train)) / float(batch_size)),
    epochs=epochs,
    verbose=2,
    validation_data=(xcv, ycv),
    callbacks=callbacks
)
