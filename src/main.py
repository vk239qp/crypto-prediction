from src.neural_network.lstm import LSTMBuilder
from src.utils.util import Util

PREV_DAYS = 30
FUTURE_DAYS = 7
LAST_SAMPLES = 1000

if __name__ == '__main__':
    util = Util()
    lstm = LSTMBuilder(PREV_DAYS, FUTURE_DAYS)

    x_train, y_train, x_test, y_test = util.prepare(PREV_DAYS, FUTURE_DAYS, LAST_SAMPLES)

    lstm.build()
    lstm.train(x_train, y_train)
    lstm.verify(x_test, y_test, util.scaler_y)
