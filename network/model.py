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


def get_model_vgg16_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    kernel_size = (3, 3)
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
    
    model = Model(input=[base_model.input, input_meta], output=output)

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
    
    model = Model(input=[base_model.input, input_meta], output=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_resnet50_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    kernel_size = (3, 3)
    optimizer = Adam(lr=0.001, decay=0.002)
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
    
    model = Model(input=[base_model.input, input_meta], output=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_mobilenet_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    kernel_size = (3, 3)
    optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #Building the model

    base_model = MobileNet(weights='imagenet', include_top=False, 
                 input_shape=input_shape, classes=1)

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
    
    model = Model(input=[base_model.input, input_meta], output=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_model_inceptionv3_pretrained(input_shape=(75, 75, 3), inputs_meta=1):
    dropout = 0.25
    kernel_size = (3, 3)
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
    
    model = Model(input=[base_model.input, input_meta], output=output)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
