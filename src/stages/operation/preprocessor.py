import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.pipeline.stage import Stage


class Preprocessor(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.test_size = self.config["preprocessor"]["data"]["test_size"]
        self.past_steps = self.config["preprocessor"]["window"]["past"]
        self.future_steps = self.config["preprocessor"]["window"]["future"]

    """
    Loading prices from csv.
    """

    def load_prices(self):
        data = pd.read_csv(f'../dataset/prices_{self.get_attributes("crypto")}.csv').tail(
            self.get_attributes("recent")).reset_index(drop=True)

        # adding custom features
        data['delta_day'] = data['high'] - data['low']
        data['prices_mean'] = data[['open', 'close', 'high', 'low']].mean(axis=1)
        data['pct_change'] = data['prices_mean'].pct_change()

        # setting time as index and converting to UTC
        data = data.set_index('time')
        data.index = pd.to_datetime(data.index, unit='s')

        return data

    """
    Loading comments from csv.
    """

    def load_comments(self):
        print("ANALYSING SENTIMENTS OF COMMENTS...")

        # dataset file name for storing or reading the data (depend on the scrapper config)
        dataset_file_name = f'../dataset/comments_dataset_{self.get_attributes("crypto")}.csv'

        # checking if we wanna load new comments or use older saved dataset
        comments_load = self.get_attributes("comments_load")

        # using old dataset
        if not comments_load:
            saved_dataset = pd.read_csv(dataset_file_name, index_col=0)
            return saved_dataset

        data = pd.read_csv(f'../dataset/comments_{self.get_attributes("crypto")}.csv').reset_index(
            drop=True)

        # converting time to UTC
        data['created_utc'] = pd.to_datetime(data['created_utc'], unit='s')

        # removing invalid comments
        data = data[data.body != '[removed]']
        data = data[data.body != '[deleted]']

        # adding sentiment columns
        analyzer = SentimentIntensityAnalyzer()
        data['compound'] = data['body'].apply(lambda body: pd.Series(analyzer.polarity_scores(str(body))))['compound']
        # grouping data by day and creating mean value from them
        data_merged = data.set_index('created_utc').groupby(pd.Grouper(freq='D')).mean().dropna()

        # adding number of comments as column
        data_merged['num_comments'] = data.set_index('created_utc').resample('D').size()

        # getting last N datas
        data_merged = data_merged[:self.get_attributes("recent")]

        # Storing last dataset
        data_merged.to_csv(dataset_file_name)

        return data_merged

    """
    Merging prices and comments datasets.
    """

    def merge_datasets(self):
        merged = pd.merge(self.load_comments(), self.load_prices(), how='inner', left_index=True, right_index=True)

        # splitting data to features and targets
        data_columns = list(merged.columns.values)
        data_columns.remove('close')
        self.add_attribute('features', len(data_columns))

        data_x = merged[data_columns]
        data_y = merged['close']

        return data_x, data_y

    """
    Preparing data for LSTM network.
    """

    def prepare(self):
        data_x, data_y = self.merge_datasets()

        # getting rid of invalid values
        data_x = data_x.fillna(0)
        data_y = data_y.fillna(0)

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
        train_x, train_y, test_x, test_y = self.prepare()
        self.add_attribute('train_x', train_x)
        self.add_attribute('train_y', train_y)
        self.add_attribute('test_x', test_x)
        self.add_attribute('test_y', test_y)

    def test(self):
        data_x, data_y = self.merge_datasets()

        data_x = data_x.fillna(0)
        data_y = data_y.fillna(0)

        test_x, test_y = self.create_window(data_x, data_y, self.past_steps, self.future_steps)

        self.add_attribute("test_x", test_x)
