import pandas
import requests
import yaml

from src.pipeline.stage import Stage
from src.utils.enums.crypto_enum import CryptoEnum


class Scrapper(Stage):
    config_path = "../src/utils/config/scrapper_config.yml"

    """    
    crypto - Cryptocurrency data which will be scrapped. Allowed formats can be found on cryptocompare web.
    """

    def __init__(self, crypto: CryptoEnum):
        with open(self.config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.url = config["cryptocompare"]["url"]
        self.token = config["cryptocompare"]["token"]
        self.crypto = crypto.value

    """
    Getting latest data from cryptocompare api.
    """

    def fetch(self):
        headers = {"Apikey": self.token}
        params = {"fsym": self.crypto, "tsym": "EUR", "allData": "true"}

        request = requests.get(url=self.url, headers=headers, params=params)

        json = request.json()['Data']['Data']

        return request.json()['Data']['Data']

    """
    Transforming json to csv file.
    """

    def transform(self, input_json, drop_columns=None):
        dataset = pandas.read_json(input_json)
        dataset.head()

    def run(self):
        json = self.fetch()
        self.transform(json)
