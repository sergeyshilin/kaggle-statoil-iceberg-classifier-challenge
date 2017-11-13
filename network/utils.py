import numpy as np


def get_data(band_1, band_2, angles, test=False):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_1])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_2])
    X_angles = np.asarray(angles).astype(np.float32)

    X_band_1, X_band_2 = remove_angle_correlation(X_band_1, X_band_2, X_angles)

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

    return data


def remove_angle_correlation(band_1, band_2, angles):
    reg_1 = np.polyfit(angles, band_1.mean(axis=(1, 2)), 1)
    reg_2 = np.polyfit(angles, band_2.mean(axis=(1, 2)), 1)

    X_band_1 = (band_1.transpose() - reg_1[0] * angles - reg_1[1]).transpose()
    X_band_2 = (band_2.transpose() - reg_2[0] * angles - reg_2[1]).transpose()

    return X_band_1, X_band_2


def get_best_history(history, monitor='val_loss', mode='min'):
    best_iteration = np.argmax(history[monitor]) if mode == 'max' else np.argmin(history[monitor])
    loss = history['loss'][best_iteration]
    acc = history['acc'][best_iteration]
    val_loss = history['val_loss'][best_iteration]
    val_acc = history['val_acc'][best_iteration]

    return best_iteration + 1, loss, acc, val_loss, val_acc
