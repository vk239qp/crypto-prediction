import csv
import os
import time
from datetime import datetime, timedelta

import requests

from src.pipeline.stage import Stage


class Scrapper(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)

        self.url = self.config["cryptocompare"]["url"]
        self.comments_url = self.config["pushshift"]["url"]
        self.token = self.config["cryptocompare"]["token"]
        self.crypto = self.config["cryptocompare"]["crypto"]
        self.drop_columns = self.config["scrapper"]["drop_columns"]
        self.load = self.config["scrapper"]["load"]

    """
    Getting latest data from cryptocompare API. Response JSON must be parsed due to further preprocessing.
    """

    def fetch_prices(self):
        headers = {"Apikey": self.token}
        params = {"fsym": self.crypto, "tsym": "EUR", "allData": "true"}

        payload = requests.get(url=self.url, headers=headers, params=params).json()

        return payload['Data']['Data']

    """
    Getting comments from reddit. Response JSON must be parsed due to further preprocessing.
    """

    def fetch_comments(self):
        n_days_ago = datetime.now() - timedelta(days=500)
        n_days_unix = time.mktime(n_days_ago.timetuple())

        subreddit = "cryptocurrency"

        if self.crypto == "ETH":
            subreddit = "ethereum"
        elif self.crypto == "BTC":
            subreddit = "bitcoin"
        # TODO pridat vilovu krypto

        params = {"sort_type": "created_utc", "sort": "asc", "after": int(n_days_unix), "size": "1000000",
                  "subreddit": subreddit}

        payload = requests.get(url=self.comments_url, params=params).json()

        return payload['data']

    """
    Transforming parsed JSON to CSV file.
    
    input_data - List of dictionaries containing data.
    """

    def convert_prices(self, input_data: list):
        drop_list = ['conversionType', 'conversionSymbol']

        if self.drop_columns is not None:
            drop_list = drop_list + self.drop_columns

        if not os.path.exists('../dataset'):
            os.makedirs('../dataset')

        with open(f'../dataset/prices_{self.crypto}.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file)

            for index, row in enumerate(input_data):
                for item in drop_list:
                    row.pop(item)

                if index == 0:
                    header = row.keys()
                    csv_writer.writerow(header)

                csv_writer.writerow(row.values())

    def convert_comments(self, input_data: list):
        if not os.path.exists('../dataset'):
            os.makedirs('../dataset')

        with open(f'../dataset/comments_{self.crypto}.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file)

            for index, row in enumerate(input_data):
                if index == 0:
                    header = row.keys()
                    csv_writer.writerow(header)

                csv_writer.writerow(row.values())

    def run(self):
        if self.load:
            self.convert_prices(self.fetch_prices())
            self.convert_comments(self.fetch_comments())

    def test(self):
        self.convert_prices(self.fetch_prices())
        self.convert_comments(self.fetch_comments())
