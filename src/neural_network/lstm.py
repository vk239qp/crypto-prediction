from keras import Sequential
from keras.layers import LSTM, Dense, Dropout


class LSTMBuilder:

    def __init__(self, back_step: int, future_step: int):
        self.model = None
        self.back_step = back_step
        self.future_step = future_step

    def build(self):
        self.model = Sequential()
        self.model.add(LSTM(30, input_shape=(self.back_step, 5), return_sequences=True))
        self.model.add(LSTM(12, return_sequences=True))
        self.model.add(LSTM(12, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(12, return_sequences=True))
        self.model.add(LSTM(12, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(12, return_sequences=True))
        self.model.add(LSTM(12, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(7))
        self.model.add(Dense(self.future_step))
        self.model.summary()
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    def train(self, data_x, data_y, epochs=800, batch=32, log_level=2):
        self.model.fit(data_x, data_y, validation_split=0.1, epochs=epochs, batch_size=batch, verbose=log_level)

    def verify(self):
        pass
