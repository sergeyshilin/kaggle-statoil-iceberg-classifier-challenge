import numpy as np


def get_data(band_1, band_2):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_1])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_2])
    X_band_3 = (X_band_1 + X_band_2) / 2
    data = np.concatenate(
        [
            X_band_1[:, :, :, np.newaxis],
            X_band_2[:, :, :, np.newaxis],
            X_band_3[:, :, :, np.newaxis]
        ],
        axis=-1
    )

    return data
