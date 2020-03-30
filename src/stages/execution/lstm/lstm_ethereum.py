from keras.layers import LSTM, Dense

from src.stages.execution.builder import Builder


class LSTMEthereum(Builder):

    def __init__(self, config_file: str):
        super().__init__(config_file)

    def build(self):
        self.model.add(LSTM(150, input_shape=(self.get_attributes('past_steps'), self.get_attributes('features')),
                            activation="tanh", return_sequences=False))
        self.model.add(Dense(self.get_attributes('future_steps')))

        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
