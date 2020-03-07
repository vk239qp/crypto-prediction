import numpy as np
import pandas as pd
from pandas import DataFrame


class Util:
    data_path = "../dataset/prices.csv"

    """
    Loading data from csv.
    
    recent - number of most recent data
    """

    def load(self, recent=1000):
        data = pd.read_csv(self.data_path).tail(recent)
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index, unit='s')

        return data

    """
    Preparing data for LSTM network.

    data - data which will be separated to features and targets
    back_step - number of days looking back
    future_step - number of days to predict
    """

    def prepare(self, back_step: int, future_step: int):
        # getting rid of invalid values
        data = self.load().replace(0, np.nan)
        data = data.fillna(data.min())

        # min max normalization
        normalized_data = (data - data.min()) / (data.max() - data.min())

        train_data, test_data = self.split(normalized_data, 0.2)

        train_x, train_y = self.create_window(train_data, back_step, future_step)
        test_x, test_y = self.create_window(test_data, back_step, future_step)

        return train_x, train_y, test_x, test_y

    """
    Window method transformation.
    
    data - data which will be separated to features and targets
    back_step - number of days looking back
    future_step - number of days to predict
    """

    def create_window(self, data: DataFrame, back_step: int, future_step: int):
        window_x, window_y = [], []

        x_data = data[['open', 'high', 'low', 'volumeTo', 'volumeFrom']]
        y_data = data['close']

        for i in range(len(data)):
            prev_days = i + back_step
            fut_days = prev_days + future_step

            if fut_days > len(data):
                break

            window_x.append(x_data.values[i:prev_days])
            window_y.append(y_data.values[prev_days:fut_days])

        return np.array(window_x), np.array(window_y)

    """
    Splitting data to train and test sets.
    
    data - data to split
    test_size - percentage of test set (from 0.0 to 1.0)
    """

    def split(self, data: DataFrame, test_size: float):
        split_row = len(data) - int(test_size * len(data))
        train_data = data.iloc[:split_row]
        test_data = data.iloc[split_row:]

        return train_data, test_data
