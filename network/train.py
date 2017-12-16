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
best_model_path = params.best_model_path

train = pd.read_json('../data/train.json')
train.loc[train['inc_angle'] == "na", 'inc_angle'] = \
    train[train['inc_angle'] != "na"]['inc_angle'].mean()

X_train, M_train = get_data(train.band_1.values, train.band_2.values, train.inc_angle.values)
X_train = params.data_adapt(X_train)
y_train = train['is_iceberg']

xtr, xcv, mtr, mcv, ytr, ycv = train_test_split(X_train, M_train, y_train, test_size=validation_split, random_state=42)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=80,
        verbose=1,
        min_delta=1e-4,
        mode='min'
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=40,
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


def get_data_generator(X, I, Y, batch_size=64):

    while True:
        # suffled indices    
        idx = np.random.permutation(X.shape[0])
        idx_0 = 0
        # create image generator
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            channel_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1
        )

        batches = datagen.flow(
            np.take(X, idx, axis=0),
            np.take(Y, idx, axis=0),
            batch_size=batch_size,
            shuffle=False
        )

        for batch in batches:
            idx_1 = idx_0 + batch[0].shape[0]

            yield [batch[0], I[ idx[ idx_0:idx_1 ] ]], batch[1]

            idx_0 = idx_1
            if idx_1 >= X.shape[0]:
                break


model = params.model_factory(input_shape=X_train.shape[1:])
model.summary()

with open(best_model_path, "w") as json_file:
    json_file.write(model.to_json())

hist = model.fit_generator(
    get_data_generator(xtr, mtr, ytr, batch_size=batch_size),
    steps_per_epoch=np.ceil(float(len(xtr)) / float(batch_size)),
    epochs=epochs,
    verbose=2,
    validation_data=get_data_generator(xcv, mcv, ycv, batch_size=batch_size),
    validation_steps=np.ceil(float(len(xcv)) / float(batch_size)),
    callbacks=callbacks
)

best_epoch, loss, acc, val_loss, val_acc = get_best_history(hist.history, monitor='val_loss', mode='min')
print ()
print ("Best epoch: {}".format(best_epoch))
print ("loss: {:0.6f} - acc: {:0.4f} - val_loss: {:0.6f} - val_acc: {:0.4f}".format(loss, acc, val_loss, val_acc))
print ()
