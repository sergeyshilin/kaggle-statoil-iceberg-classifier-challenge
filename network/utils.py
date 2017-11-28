import numpy as np


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

    # data = color_composite(data)

    return data, X_meta


def color_composite(data):
    rgb_arrays = np.zeros(data.shape).astype(np.float32)
    for i, data_row in enumerate(data):
        band_1 = data_row[:,:,0]
        band_2 = data_row[:,:,1]
        band_3 = data_row[:,:,2]

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

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
