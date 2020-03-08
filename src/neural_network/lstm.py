from datetime import datetime

from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout


class LSTMBuilder:

    def __init__(self, back_step: int, future_step: int):
        self.model = None
        self.features = 5
        self.back_step = back_step
        self.future_step = future_step

    # building model
    def build(self):
        self.model = Sequential()
        self.model.add(
            LSTM(100, input_shape=(self.back_step, self.features), activation="softsign", return_sequences=True))
        self.model.add(LSTM(80, activation="softsign", return_sequences=True))
        self.model.add(LSTM(60, activation="softsign", return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(60, activation="softsign", return_sequences=True))
        self.model.add(LSTM(60, activation="softsign", return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(30, activation="softsign"))
        self.model.add(Dense(self.future_step))
        self.model.summary()
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # training model
    def train(self, data_x, data_y, epochs=600, batch=64, log_level=2):
        results = self.model.fit(data_x, data_y, validation_split=0.1, epochs=epochs, batch_size=batch,
                                 verbose=log_level)

        # plotting loss
        history = results.history
        plt.figure(figsize=(12, 4))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        # plotting accuracy
        plt.figure(figsize=(12, 4))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

    def verify(self, data_x: np.array, data_y: np.array, scaler: MinMaxScaler):
        plt.figure(figsize=(12, 5))
        # getting time to add as timestamp to graph
        date_time = datetime.now()
        date_time_formatted = date_time.strftime("%d/%m/%Y-%H:%M")

        # Getting predictions by predicting from the last X
        prediction = self.model.predict(data_x[-1].reshape(1, self.back_step, self.features)).tolist()[0]

        # Transforming normalized values back to normal range
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

        # Getting the actual values from the last available y variable which correspond to its respective X variable
        actual = scaler.inverse_transform(data_y[-1].reshape(-1, 1))

        # Printing and plotting those predictions
        print("Predicted Prices:\n", prediction)
        plt.plot(prediction, label='Predicted')

        # Printing and plotting the actual values
        print("\nActual Prices:\n", actual.tolist())
        plt.plot(actual.tolist(), label='Actual')

        plt.title(f"Predicted vs Actual Closing Prices")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        # plt.savefig(f"../graphs/result-{date_time_formatted}.png")
