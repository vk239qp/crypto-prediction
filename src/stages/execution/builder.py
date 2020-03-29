import os
from abc import abstractmethod
from datetime import datetime

from keras import Sequential
from pandas import np

from src.pipeline.stage import Stage
from src.stages.execution.plotter import Plotter


class Builder(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)

        self.batch_size = self.config["network"]["batch"]
        self.epochs = self.config["network"]["epochs"]
        self.log_level = self.config["network"]["log"]
        self.validation = self.config["network"]["validation"]

        self.model = Sequential()
        self.plotter = Plotter()

    """
    Building model 
    """

    @abstractmethod
    def build(self):
        pass

    """
    Training model
    """

    def train(self):
        results = self.model.fit(self.get_attributes('train_x'),
                                 self.get_attributes('train_y'),
                                 validation_split=self.validation,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=self.log_level, shuffle=False)

        # plotting loss
        history = results.history
        self.plotter.show(data=[history['val_loss'], history['loss']],
                          legend=['val_loss', 'loss'],
                          title='Loss',
                          x_label='Epochs',
                          y_label='Loss')

        # plotting accuracy
        self.plotter.show(data=[history['val_accuracy'], history['accuracy']],
                          legend=['val_accuracy', 'accuracy'],
                          title='Accuracy',
                          x_label='Epochs',
                          y_label='Accuracy')

    """
    Verifying model
    """

    def verify(self):
        # getting time to add as timestamp to graph
        date_time = datetime.now()
        date_time_formatted = date_time.strftime("%d/%m/%Y-%H:%M")

        scaler = self.get_attributes('scaler_y')
        test_x = self.get_attributes('test_x')
        test_y = self.get_attributes('test_y')
        past_steps = self.get_attributes('past_steps')
        features = self.get_attributes('features')

        # Getting predictions by predicting from the last X
        prediction = self.model.predict(test_x[-1].reshape(1, past_steps, features)).tolist()[0]

        # Transforming normalized values back to normal range
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

        # Getting the actual values from the last available y variable which correspond to its respective X variable
        actual = scaler.inverse_transform(test_y[-1].reshape(-1, 1))

        print("Predicted Prices:\n", prediction.tolist())
        print("Actual Prices:\n", actual.tolist())

        self.plotter.show(data=[prediction, actual],
                          legend=['Predicted', 'True'],
                          title="Closing Prices",
                          x_label="Day",
                          x_ticks=1.0,
                          y_label="Price")

    def save(self):
        date_time = datetime.now()
        date_time_formatted = date_time.strftime("%d-%m-%Y-%H:%M")

        if not os.path.exists("../results/model"):
            os.makedirs("../results/model")

        with open(f'../results/model/lstm_{self.get_attributes("crypto")}_{date_time_formatted}.h5', "wb") as file:
            self.model.save(file)

    def run(self):
        self.build()
        self.train()
        self.verify()
        self.save()
