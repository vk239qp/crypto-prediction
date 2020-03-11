import numpy as np
from abc import ABC, abstractmethod

from keras import Sequential


class Builder(ABC):
    """
    Default constructor for LSTM network

    features - Number of features in dataset
    back_step - Number of days to be considered for prediction
    future_step - Number of days to predict prices
    """

    def __init__(self, features: int, back_step: int, future_step: int):
        self.model = Sequential()
        self.features = features
        self.back_step = back_step
        self.future_step = future_step

    """
    Building model 
    """

    @abstractmethod
    def build(self):
        pass

    """
    Training model
    
    data_x - Numpy array of features
    data_y - Numpy array of targets
    epochs - Number of training epochs
    batch - Batch size
    log_level - Log level
    """

    # def train(self, data_x: np.array, data_y: np.array, epochs=500, batch=32, log_level=2):
    #     results = self.model.fit(data_x, data_y, validation_split=0.1, epochs=epochs, batch_size=batch,
    #                              verbose=log_level)
    #
    #     # plotting loss
    #     history = results.history
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(history['val_loss'])
    #     plt.plot(history['loss'])
    #     plt.legend(['val_loss', 'loss'])
    #     plt.title('Loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.show()
    #
    #     # plotting accuracy
    #     plt.figure(figsize=(12, 4))
    #     plt.plot(history['val_accuracy'])
    #     plt.plot(history['accuracy'])
    #     plt.legend(['val_accuracy', 'accuracy'])
    #     plt.title('Accuracy')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.show()
    #
    # """
    # Verifying model
    # """
    #
    # def verify(self, data_x: np.array, data_y: np.array, scaler: MinMaxScaler):
    #     plt.figure(figsize=(12, 5))
    #     # getting time to add as timestamp to graph
    #     date_time = datetime.now()
    #     date_time_formatted = date_time.strftime("%d/%m/%Y-%H:%M")
    #
    #     # Getting predictions by predicting from the last X
    #     prediction = self.model.predict(data_x[-1].reshape(1, self.back_step, self.features)).tolist()[0]
    #
    #     # Transforming normalized values back to normal range
    #     prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    #
    #     # Getting the actual values from the last available y variable which correspond to its respective X variable
    #     actual = scaler.inverse_transform(data_y[-1].reshape(-1, 1))
    #
    #     # Printing and plotting those predictions
    #     print("Predicted Prices:\n", prediction.tolist())
    #     plt.plot(prediction, label='Predicted')
    #
    #     # Printing and plotting the actual values
    #     print("\nActual Prices:\n", actual.tolist())
    #     plt.plot(actual.tolist(), label='Actual')
    #
    #     plt.title(f"Predicted vs Actual Closing Prices")
    #     plt.ylabel("Price")
    #     plt.legend()
    #     plt.show()
    #     # plt.savefig(f"../graphs/result-{date_time_formatted}.png")
