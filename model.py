from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Add, GlobalMaxPooling2D
from keras.layers import Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD, Adadelta


def get_model_sequential(input_shape=(75, 75, 3)):
    dropout = 0.25
    kernel_size = (3, 3)
    optimizer = Adam(lr=0.001)
    #Building the model
    model = Sequential()
    # size = 75x75

    #Conv Layer 1
    model.add(Conv2D(32, kernel_size=kernel_size, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))
    # size = 37x37

    #Conv Layer 2
    model.add(Conv2D(64, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))
    # size = 18x18

    #Conv Layer 3
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout))
    # size = 9x9

    #Conv Layer 3
    model.add(Conv2D(256, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(dropout))
    # size = 3x3

    #Conv Layer 4
    model.add(Conv2D(512, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, kernel_size=kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(dropout))
    # size = 1x1x512

    #Flatten the data for upcoming dense layers
    model.add(Flatten())

    #Dense Layers
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    #Dense Layer 2
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    #Sigmoid Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def get_model_residual(input_shape=(75, 75, 3)):
    dropout = 0.25
    kernel_size = (5, 5)
    # optimizer = optimizer=SGD(lr=0.001, momentum=0.9)
    optimizer = Adam(lr=0.001)

    inputs = Input(shape=input_shape)
    inputs_norm = BatchNormalization()(inputs)

    #Conv Layer 1
    conv1 = Conv2D(32, kernel_size=kernel_size, padding='same')(inputs_norm)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv1 = Dropout(dropout)(conv1)

    conv1 = Conv2D(64, kernel_size=kernel_size, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv1 = Dropout(dropout)(conv1)

    # Conv Layer Residual
    conv1_residual = BatchNormalization()(conv1)
    conv1_residual = Activation('relu')(conv1_residual)
    conv1_residual = Conv2D(128, kernel_size=kernel_size, padding='same')(conv1_residual)
    conv1_residual = BatchNormalization()(conv1_residual)
    conv1_residual = Activation('relu')(conv1_residual)
    conv1_residual = Dropout(dropout)(conv1_residual)
    conv1_residual = Conv2D(64, kernel_size=kernel_size, padding='same')(conv1_residual)
    conv1_residual = BatchNormalization()(conv1_residual)
    conv1_residual = Activation('relu')(conv1_residual)

    conv1_residual = Add()([conv1_residual, conv1])

    #Conv Layer 2
    conv2 = Conv2D(128, kernel_size=kernel_size, padding='same')(conv1_residual)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv2 = Dropout(dropout)(conv2)

    conv2 = Conv2D(256, kernel_size=kernel_size, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv2 = Dropout(dropout)(conv2)

    conv2 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv2 = Dropout(dropout)(conv2)
    conv2 = GlobalMaxPooling2D()(conv2)

    #Dense Layers
    fc1 = Dense(512)(conv2)
    fc1 = BatchNormalization()(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(dropout)(fc1)

    #Dense Layer 2
    fc2 = Dense(256)(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = Dropout(dropout)(fc2)

    #Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
