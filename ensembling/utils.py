import numpy as np
import pandas as pd

train = pd.read_json('../data/train.json')
test = pd.read_json('../data/test.json')

def get_object_size(arr):
    p = np.reshape(np.array(arr), [75, 75]) > (np.mean(np.array(arr)) + 2 * np.std(np.array(arr)))
    iso = p * np.reshape(np.array(arr), [75, 75])
    return np.sum(iso < -5)

def get_stats(data, inc_angle):
    if inc_angle:
        data.loc[data['inc_angle'] == "na", 'inc_angle'] = 0
    for i in range(2):
        label = str(i + 1)
        band_name = 'band_' + label
        
        bands = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data[band_name].values])
        data['size_' + label] = [get_object_size(x) for x in bands]
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

print(train.columns.values, test.columns.values)

train = get_stats(train, True)
test = get_stats(test, False)

train.drop(['band_1', 'band_2'], axis = 1).to_csv('train_metadata.csv', index = False)
test.drop(['band_1', 'band_2'], axis = 1).to_csv('test_metadata.csv', index = False)
