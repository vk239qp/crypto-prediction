import yaml


class Scrapper:

    def __init__(self):
        with open('scrapper_config.yaml', 'r') as file:
            config = yaml.load(file)
            print(config["cryptocompare"])
