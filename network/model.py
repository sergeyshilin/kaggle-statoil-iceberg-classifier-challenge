from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Add, GlobalMaxPooling2D
from keras.layers import Concatenate, concatenate
from keras.layers import Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD, Adadelta

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception


def get_model_vgg16_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = VGG16(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)
    
    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_vgg16(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    optimizer = Adam(lr=1e-3, decay=1e-5)
    #Building the model

    base_model = VGG16(weights=None, include_top=False,
                 input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(256, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_vgg19_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.3
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = VGG19(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)
    
    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_resnet50_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = ResNet50(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('avg_pool').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_mobilenet_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = MobileNet(weights='imagenet', include_top=False, 
                 dropout=0.2, input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('conv_pw_13_relu').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_inceptionv3_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = InceptionV3(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('mixed10').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_xception_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = Xception(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    x = base_model.get_layer('block14_sepconv2_act').output

    x = GlobalMaxPooling2D()(x)
    concat = concatenate([x, input_meta_norm])
    fc1 = Dense(512, activation='relu', name='fc2')(concat)
    fc1 = Dropout(dropout)(fc1)
    fc2 = Dense(512, activation='relu', name='fc3')(fc1)
    fc2 = Dropout(dropout)(fc2)

    # Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[base_model.input, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_custom(input_shape=(75, 75, 3), inputs_meta=1):
    kernel_size = (3, 3)
    optimizer = Adam(lr=0.01, decay=0.0)

    input_bands = Input(shape=input_shape, name='bands')
    inputs_bands_norm = BatchNormalization()(input_bands)

    input_meta = Input(shape=[inputs_meta], name='meta')
    input_meta_norm = BatchNormalization()(input_meta)

    # Conv Layer 1
    conv1 = Conv2D(64, kernel_size=kernel_size)(inputs_bands_norm)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)
    conv1 = Dropout(0.2)(conv1)
    # size = 25x25
 
    # Conv Layer 2
    conv2 = Conv2D(128, kernel_size=kernel_size)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    conv2 = Dropout(0.2)(conv2)
    # size = 12x12

    conv3 = Conv2D(128, kernel_size=kernel_size)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    conv3 = Dropout(0.3)(conv3)
    # size = 5x5

    # Conv Layer 4
    conv4 = Conv2D(64, kernel_size=kernel_size)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
    conv4 = Dropout(0.3)(conv4)
    # size = 2x2

    conv4 = GlobalMaxPooling2D()(conv4)
    conv4 = BatchNormalization()(conv4)

    concat = Concatenate()([conv4, input_meta_norm])

    #Dense Layers
    fc1 = Dense(512)(concat)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(0.2)(fc1)

    #Dense Layer 2
    fc2 = Dense(256)(fc1)
    fc2 = Activation('relu')(fc2)
    fc2 = Dropout(0.2)(fc2)

    #Sigmoid Layer
    output = Dense(1)(fc2)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input_bands, input_meta], outputs=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
