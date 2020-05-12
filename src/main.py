from src.helpers.CryptoCurrency import CryptoCurrency
from src.helpers.Runner import Runner

if __name__ == '__main__':
    # 0 - train, 1 - test, 2 - comparing
    mode = 0

    crypto_currencies_to_run = [
        CryptoCurrency("Bitcoin", "BTC")
        # CryptoCurrency("Ethereum", "ETH")
    ]
    Runner(crypto_currencies_to_run, mode)
