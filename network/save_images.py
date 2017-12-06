import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMAGES_TO_SAVE = 300

import numpy as np
import pandas as pd

import params
from utils import get_data, get_best_history
from utils import preprocess_image

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

batch_size = params.batch_size

train = pd.read_json('../data/train.json')
test = pd.read_json('../data/test.json')

train.loc[train['inc_angle'] == "na", 'inc_angle'] = \
    train[train['inc_angle'] != "na"]['inc_angle'].mean()

X_train, _ = get_data(train.band_1.values, train.band_2.values, train.inc_angle.values)
X_test, _ = get_data(test.band_1.values, test.band_2.values, test.inc_angle.values)


datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    channel_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    preprocessing_function=preprocess_image
)

datagen_test = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    channel_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0
)

for i, batch in enumerate(datagen.flow(X_train, batch_size=batch_size,
                          save_to_dir='../preview', save_format='jpeg')):
    if i > IMAGES_TO_SAVE:
        break

# Save test images to display
for i, batch in enumerate(datagen_test.flow(X_test, batch_size=batch_size,
                          save_to_dir='../preview_test', save_format='jpeg')):
    if i > IMAGES_TO_SAVE:
        break
