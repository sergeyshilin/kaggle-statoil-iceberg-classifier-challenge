import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd

import params
from utils import get_data, get_best_history

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

epochs = params.max_epochs
batch_size = params.batch_size
validation_split = params.validation_split
best_weights_path = params.best_weights_path

train = pd.read_json('data/train.json')
train.loc[train['inc_angle'] == "na", 'inc_angle'] = \
    train[train['inc_angle'] != "na"]['inc_angle'].mean()

X_train = get_data(train.band_1.values, train.band_2.values, train.inc_angle.values)
y_train = train['is_iceberg']

xtr, xcv, ytr, ycv = train_test_split(X_train, y_train, test_size=validation_split, random_state=42)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=40,
        verbose=1,
        min_delta=1e-4,
        mode='min'
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=20,
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

model = params.model_factory(input_shape=X_train.shape[1:])
model.summary()

hist = model.fit_generator(
    datagen.flow(xtr, ytr, batch_size=batch_size),
    steps_per_epoch=np.ceil(float(len(X_train)) / float(batch_size)),
    epochs=epochs,
    verbose=2,
    validation_data=(xcv, ycv),
    callbacks=callbacks
)

best_epoch, loss, acc, val_loss, val_acc = get_best_history(hist.history, monitor='val_loss', mode='min')
print ()
print ("Best epoch: {}".format(best_epoch))
print ("loss: {:0.6f} - acc: {:0.4f} - val_loss: {:0.6f} - val_acc: {:0.4f}".format(loss, acc, val_loss, val_acc))
print ()
