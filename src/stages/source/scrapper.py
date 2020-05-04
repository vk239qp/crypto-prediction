import csv
import os

import requests

from src.pipeline.stage import Stage


class Scrapper(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)

        self.url = self.config["cryptocompare"]["url"]
        self.token = self.config["cryptocompare"]["token"]
        self.crypto = self.config["cryptocompare"]["crypto"]
        self.drop_columns = self.config["scrapper"]["drop_columns"]
        self.load = self.config["scrapper"]["load"]

    """
    Getting latest data from cryptocompare API. Response JSON must be parsed due to further preprocessing.
    """

    def fetch(self):
        headers = {"Apikey": self.token}
        params = {"fsym": self.crypto, "tsym": "EUR", "allData": "true"}

        payload = requests.get(url=self.url, headers=headers, params=params).json()

        return payload['Data']['Data']

    """
    Transforming parsed JSON to CSV file.
    
    input_data - List of dictionaries containing data.
    """

    def convert(self, input_data: list):
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

    def run(self):
        if self.load:
            json_data = self.fetch()
            self.convert(json_data)

    def test(self):
        self.convert(self.fetch())
