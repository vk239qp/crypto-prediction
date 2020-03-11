import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from src.pipeline.stage import Stage


class Preprocessor(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.recent = self.config["preprocessor"]["data"]["recent"]
        self.strategy = self.config["preprocessor"]["data"]["strategy"]
        self.test_size = self.config["preprocessor"]["data"]["test_size"]
        self.past_steps = self.config["preprocessor"]["window"]["past"]
        self.future_steps = self.config["preprocessor"]["window"]["future"]

    """
    Loading data from csv.
    """

    def load(self):
        data = pd.read_csv(f'../dataset/prices_{self.get_attributes("crypto")}.csv').tail(self.recent)
        data = data.set_index('time')
        data.index = pd.to_datetime(data.index, unit='s')

        data_columns = list(data.columns.values)
        data_columns.remove('close')

        data_x = data[data_columns]
        data_y = data['close']

        return data_x, data_y

    """
    Preparing data for LSTM network.
    """

    def prepare(self):
        data_x, data_y = self.load()

        # getting rid of invalid values
        data_x = data_x.replace(0, np.nan).fillna(data_x.min())
        data_y = data_y.replace(0, np.nan).fillna(data_y.min())

        train_x, train_y, test_x, test_y = self.split(data_x, data_y, self.test_size)

        train_x, train_y = self.create_window(train_x, train_y, self.past_steps, self.future_steps)
        test_x, test_y = self.create_window(test_x, test_y, self.past_steps, self.future_steps)

        return train_x, train_y, test_x, test_y

    """
    Window method transformation.
    
    data_x - features which will be separated to time windows
    data_y - targets which will be separated to time windows
    past_steps - number of days looking back
    future_steps - number of days to predict
    """

    def create_window(self, data_x: DataFrame, data_y: DataFrame, past_steps: int, future_steps: int):
        window_x, window_y = [], []

        # min max scaling
        normalized_x = pd.DataFrame(self.scaler_x.fit_transform(data_x), columns=data_x.columns, index=data_x.index)
        normalized_y = pd.DataFrame(self.scaler_y.fit_transform(data_y.values.reshape(-1, 1))).values.flatten()

        for i in range(len(data_x)):
            prev_days = i + past_steps
            fut_days = prev_days + future_steps

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

    def run(self):
        self.prepare()
