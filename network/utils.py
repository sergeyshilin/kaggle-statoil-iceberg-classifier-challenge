import numpy as np
import cv2

from keras.preprocessing import image


def get_data(band_1, band_2, *meta):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_1])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_2])

    X_band_1_val = np.power(10, X_band_1 / 20.0)
    X_band_2_val = np.power(10, X_band_2 / 20.0)

    X_band_sqrmean = 20 * np.log10(np.sqrt(np.square(X_band_1_val) + np.square(X_band_2_val)))

    data = np.concatenate(
        [
            X_band_1[:, :, :, np.newaxis],
            X_band_2[:, :, :, np.newaxis],
            X_band_sqrmean[:, :, :, np.newaxis]
        ],
        axis=-1
    )

    X_meta = np.concatenate(
        [
            np.asarray(m).astype(np.float32)[..., np.newaxis] for m in meta
        ], 
        axis=1
    )

    # data = data_normalization(data)

    return data, X_meta

def identical(data):
    return data

def data_normalization(data):
    rgb_arrays = np.zeros(data.shape).astype(np.float32)
    for i, data_row in enumerate(data):
        band_1 = data_row[:,:,0]
        band_2 = data_row[:,:,1]
        band_3 = data_row[:,:,2]

        r = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        rgb_arrays[i] = rgb
    return np.array(rgb_arrays)


def get_best_history(history, monitor='val_loss', mode='min'):
    best_iteration = np.argmax(history[monitor]) if mode == 'max' else np.argmin(history[monitor])
    loss = history['loss'][best_iteration]
    acc = history['acc'][best_iteration]
    val_loss = history['val_loss'][best_iteration]
    val_acc = history['val_acc'][best_iteration]

    return best_iteration + 1, loss, acc, val_loss, val_acc


# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def get_data_generator(datagen, X1, X2, y, batch_size):
    genX1 = datagen.flow(X1, y, batch_size=batch_size, seed=55)
    genX2 = datagen.flow(X1, X2, batch_size=batch_size, seed=55)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()

        #Assert arrays are equal - this was for peace of mind, but slows down training
        #np.testing.assert_array_equal(X1i[0],X2i[0])
        yield [X1i[0], X2i[1]], X1i[1]


def get_data_generator_test(datagen, X1, X2, batch_size):
    genX1 = datagen.flow(X1, X2, batch_size=batch_size, shuffle=False)

    while True:
        X1i = genX1.next()
        yield [X1i[0], X1i[1]]

def get_object_size(arr):
    p = np.reshape(np.array(arr), [75, 75]) > (np.mean(np.array(arr)) + 2 * np.std(np.array(arr)))
    iso = p * np.reshape(np.array(arr), [75, 75])
    return np.sum(iso < -5)

def get_stats(data):
    for i in range(2):
        label = str(i + 1)
        band_name = 'band_' + label
        bands = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data[band_name].values])

        data['max_' + label] = [np.max(x) for x in bands]
        data['maxpos_' + label] = [np.argmax(x) for x in bands]
        data['min_' + label] = [np.min(x) for x in bands]
        data['minpos_' + label] = [np.argmin(x) for x in bands]
        data['med_' + label] = [np.median(x) for x in bands]
        data['std_' + label] = [np.std(x) for x in bands]
        data['mean_' + label] = [np.mean(np.array(x)) for x in bands]
        data['p25_' + label] = [np.sort(x.reshape(75 * 75))[int(0.25 * 75 * 75)] for x in bands]
        data['p75_' + label] = [np.sort(x.reshape(75 * 75))[int(0.75 * 75 * 75)] for x in bands]
        data['mid50_' + label] = data['p75_' + label] - data['p25_' + label]

    return data

def resize_data(data, size):
    data_upscaled = np.zeros((data.shape[0], size[0], size[1], size[2]), dtype=data.dtype)

    for i in range(len(data)):
        data_upscaled[i] = cv2.resize(data[i], (size[0], size[1]))

    return data_upscaled[:]


## Augmentation methods


def random_rotate(img, angle=10, u=0.5):
    if np.random.random() < u:
        img = image.random_rotation(img, angle, row_axis=0, col_axis=1, channel_axis=2)
    return img

def random_shift(img, w_limit=0.1, h_limit=0.1, u=0.5):
    if np.random.random() < u:
        img = image.random_shift(img, w_limit, h_limit, row_axis=0, col_axis=1, channel_axis=2)
    return img

def random_zoom(img, zoom=0.1, u=0.5):
    if np.random.random() < u:
        img = image.random_zoom(img, (1 - zoom, 1 + zoom), row_axis=0, col_axis=1, channel_axis=2)
    return img

def convert_to_gray(img):
    b, g, r = cv2.split(img)
    coef = np.array([[[0.114, 0.587, 0.299]]])  # bgr to gray (YCbCr)
    gray = np.sum(img * coef, axis=2)
    img = np.dstack((b-2, b-2, r))
    return img

def generate_test_like_image(img):
    shift = 20
    # flip image horizontally and mirror left part on the right part
    img = img[::-1, :, : ]
    img = np.roll(img, shift=shift, axis=0)
    img[:37, :, : ] = img[38:, :, : ]

    # the same vertically
    img = img[:, ::-1, : ]
    img = np.roll(img, shift=shift, axis=1)
    img[:, :37, : ] = img[:, 38:, : ]
    return img

def half_image_gamma_correction(img):
    img[38:] = img[38:] + np.random.choice(np.arange(3, 11))
    return img

def preprocess_image(image):
    image_aug = image[:]
    transformation_proba = 0.5

    if np.random.random() < transformation_proba:
        # Generate an image like we have machine-generated data in a test dataset
        image_aug = generate_test_like_image(image_aug)
        image_aug = convert_to_gray(image_aug)
        image_aug = half_image_gamma_correction(image_aug)
        image_aug = random_shift(image_aug, w_limit=0.2, h_limit=0.2, u=0.7)
        image_aug = random_rotate(image_aug, angle=30, u=0.9)
    else:
        pass
        image_aug = random_zoom(image_aug, zoom=0.4, u=0.5)
        image_aug = random_rotate(image_aug, angle=10, u=0.5)

    return image_aug
