import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from src.utils.scrapper import Scrapper


class DataUtil:
    data_path = "../dataset/prices.csv"

    def __init__(self):
        self.scrapper = Scrapper()
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    """
    Loading data from csv.
    
    recent - number of most recent data
    """

    def load(self, recent: int):
        data = pd.read_csv(self.data_path).tail(recent)
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index, unit='s')

        data_x = data[['open', 'high', 'low', 'volumeFrom', 'volumeTo']]
        data_y = data['close']

        return data_x, data_y

    """
    Preparing data for LSTM network.

    data - data which will be separated to features and targets
    back_step - number of days looking back
    future_step - number of days to predict
    """

    def prepare(self, back_step: int, future_step: int, last_samples=None):
        data_x, data_y = self.load(last_samples)

        # getting rid of invalid values
        data_x = data_x.replace(0, np.nan).fillna(data_x.min())
        data_y = data_y.replace(0, np.nan).fillna(data_y.min())

        train_x, train_y, test_x, test_y = self.split(data_x, data_y, 0.2)

        train_x, train_y = self.create_window(train_x, train_y, back_step, future_step)
        test_x, test_y = self.create_window(test_x, test_y, back_step, future_step)

        return train_x, train_y, test_x, test_y

    """
    Window method transformation.
    
    data_x - features which will be separated to time windows
    data_y - targets which will be separated to time windows
    back_step - number of days looking back
    future_step - number of days to predict
    """

    def create_window(self, data_x: DataFrame, data_y: DataFrame, back_step: int, future_step: int):
        window_x, window_y = [], []

        # min max scaling
        normalized_x = pd.DataFrame(self.scaler_x.fit_transform(data_x), columns=data_x.columns, index=data_x.index)
        normalized_y = pd.DataFrame(self.scaler_y.fit_transform(data_y.values.reshape(-1, 1))).values.flatten()

        for i in range(len(data_x)):
            prev_days = i + back_step
            fut_days = prev_days + future_step

            if fut_days > len(data_x):
                break
            window_x.append(normalized_x.values[i:prev_days])
            window_y.append(normalized_y[prev_days:fut_days])

        return np.array(window_x), np.array(window_y)

    """
    Splitting data to train and test sets.
    
    data_x - features to split
    data_y - targets to split
    test_size - percentage of test set (from 0.0 to 1.0)
    """

    def split(self, data_x: DataFrame, data_y: DataFrame, test_size: float):
        split_row = len(data_x) - int(test_size * len(data_x))

        train_x = data_x.iloc[:split_row]
        test_x = data_x.iloc[split_row:]

        train_y = data_y.iloc[:split_row]
        test_y = data_y.iloc[split_row:]

        return train_x, train_y, test_x, test_y
