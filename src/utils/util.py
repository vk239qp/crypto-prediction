import numpy as np
import pandas as pd


class Util:
    data_path = "../dataset/prices.csv"

    # loading data from csv
    def load(self):
        data = pd.read_csv(self.data_path)
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index, unit='s')

        return data

    # preparing data for LSTM network
    def prepare(self):
        data = self.load().replace(0, np.nan)
        data = data.fillna(data.min())

        # min max normalization
        normalized_data = (data - data.min()) / (data.max() - data.min())

        return self.split(normalized_data, 0.2)

    # splitting data to train and test sets
    def split(self, data, test_size):
        split_row = len(data) - int(test_size * len(data))
        train_data = data.iloc[:split_row]
        test_data = data.iloc[split_row:]

        x_train = train_data[['open', 'high', 'low', 'volumeTo', 'volumeFrom']]
        y_train = train_data['close']

        x_test = test_data[['open', 'high', 'low', 'volumeTo', 'volumeFrom']]
        y_test = test_data['close']

        return x_train, x_test, y_train, y_test
