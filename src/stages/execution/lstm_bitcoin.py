from keras.layers import LSTM, Dense

from src.stages.execution.builder import Builder


class LSTMBitcoin(Builder):

    def __init__(self, features: int, back_step: int, future_step: int):
        super(LSTMBitcoin, self).__init__(features, back_step, future_step)

    def build(self):
        self.model.add(LSTM(120, input_shape=(self.back_step, self.features), activation="relu", return_sequences=True))
        self.model.add(LSTM(60, activation="relu", return_sequences=True))
        self.model.add(LSTM(60, activation="relu", return_sequences=True))
        self.model.add(LSTM(30, activation="relu"))
        self.model.add(Dense(self.future_step))

        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
