import pandas as pd
import numpy as np

threshold = 0.03
test = pd.read_json('../data/test.json')
probas = pd.read_csv('../submits/submission_056.csv').is_iceberg.values

is_real = test.inc_angle.apply(str).apply(lambda x: len(x.split('.')[1]) > 7)
is_iceberg = np.zeros((test.shape[0])) - 1
is_iceberg[probas > 1.0 - threshold] = 1
is_iceberg[probas < threshold] = 0
is_iceberg[is_real] = -1

test['is_iceberg'] = pd.Series(is_iceberg, index = test.index)
test.to_json('../data/test.json')
