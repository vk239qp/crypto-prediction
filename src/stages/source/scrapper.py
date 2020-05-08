import csv
import os
import time
from datetime import datetime, timedelta

import requests

from src.pipeline.stage import Stage


class Scrapper(Stage):

    def __init__(self, config_file: str, crypto: str):
        super().__init__(config_file)

        self.crypto = crypto

        self.load = self.config["scrapper"]["load"]
        self.recent = self.config["scrapper"]["recent"]
        self.prices_url = self.config["cryptocompare"]["url"]
        self.token = self.config["cryptocompare"]["token"]
        self.comments_load = self.config["pushshift"]["load"]
        self.comments_url = self.config["pushshift"]["url"]

    """
    Getting latest data from cryptocompare API. Response JSON must be parsed due to further preprocessing.
    """

    def fetch_prices(self):
        print("SCRAPPING PRICES...")

        headers = {"Apikey": self.token}
        params = {"fsym": self.crypto, "tsym": "EUR", "allData": "true"}

        payload = requests.get(url=self.prices_url, headers=headers, params=params).json()

        return payload['Data']['Data']

    """
    Getting comments from reddit. Response JSON must be parsed due to further preprocessing.
    """

    def fetch_comments(self):
        print("SCRAPPING COMMENTS...")

        subreddit = "cryptocurrency"

        if self.crypto == "ETH":
            subreddit = "ethereum"
        elif self.crypto == "BTC":
            subreddit = "bitcoin"
        elif self.crypto == "XTZ":
            subreddit = "tezos"

        days = self.recent
        payload = []

        while days != 0:
            time.sleep(1)
            n_days_ago = datetime.now() - timedelta(days=days)
            n_days_unix = time.mktime(n_days_ago.timetuple())

            params = {"sort_type": "created_utc", "sort": "asc", "after": int(n_days_unix), "size": 1000,
                      "subreddit": subreddit}

            payload.append(requests.get(url=self.comments_url, params=params).json()['data'])
            days -= 1

        return payload

    """
    Transforming parsed JSON of prices to CSV file.
    
    input_data - List of dictionaries containing data.
    """

    def convert_prices(self, input_data: list):
        drop_list = ['conversionType', 'conversionSymbol']

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

    """
    Transforming parsed JSON of comments to CSV file.

    input_data - List of dictionaries containing data.
    """

    def convert_comments(self, input_data: list):
        if not os.path.exists('../dataset'):
            os.makedirs('../dataset')

        with open(f'../dataset/comments_{self.crypto}.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file)

            for index, _ in enumerate(input_data):
                if index == 0:
                    csv_writer.writerow(['created_utc', 'body', 'score'])
                for _, row in enumerate(input_data[index]):
                    values = (row['created_utc'], row['body'], row['score'])
                    csv_writer.writerow(values)

    def run(self):
        if self.load:
            self.convert_prices(self.fetch_prices())
            self.convert_comments(self.fetch_comments())

    def test(self):
        if self.load:
            self.convert_prices(self.fetch_prices())
            self.convert_comments(self.fetch_comments())
