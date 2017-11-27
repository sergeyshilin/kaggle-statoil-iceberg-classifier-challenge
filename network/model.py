from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Add, GlobalMaxPooling2D, Concatenate
from keras.layers import Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD, Adadelta


def get_model_sequential(input_shape=(75, 75, 3)):
    dropout = 0.25
    dropout_fc = 0.5
    kernel_size = (3, 3)
    optimizer = Adam(lr=0.001, decay=0.002)
    #Building the model
    
    input_bands = Input(shape=input_shape, name='bands')
    inputs_bands_norm = BatchNormalization()(input_bands)

    input_angle = Input(shape=[1], name='angle')
    input_angle_norm = BatchNormalization()(input_angle)

    # Conv Layer 1
    conv1 = Conv2D(64, kernel_size=kernel_size, padding='same')(inputs_bands_norm)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, kernel_size=kernel_size, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv1 = Dropout(dropout)(conv1)
    # size = 37x37
 
    # Conv Layer 2
    conv2 = Conv2D(128, kernel_size=kernel_size, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, kernel_size=kernel_size, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv2 = Dropout(dropout)(conv2)
    # size = 18x18

    # Conv Layer 3
    conv3 = Conv2D(256, kernel_size=kernel_size, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, kernel_size=kernel_size, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, kernel_size=kernel_size, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, kernel_size=kernel_size, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    conv3 = Dropout(dropout)(conv3)
    # size = 9x9

    # Conv Layer 4
    conv4 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv4)
    conv4 = Dropout(dropout)(conv4)
    # size = 3x3

    # Conv Layer 5
    conv5 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, kernel_size=kernel_size, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(conv5)
    conv5 = Dropout(dropout)(conv5)
    # size = 1x1x512

    conv5 = GlobalMaxPooling2D()(conv5)
    conv5 = BatchNormalization()(conv5)

    concat = Concatenate()([conv5, input_angle_norm])

    #Dense Layers
    fc1 = Dense(512)(concat)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(dropout_fc)(fc1)

    #Dense Layer 2
    fc2 = Dense(256)(fc1)
    fc2 = Activation('relu')(fc2)
    fc2 = Dropout(dropout_fc)(fc2)

    #Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input_bands, input_angle], outputs=output)

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
