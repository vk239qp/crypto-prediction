import os
from abc import abstractmethod
from datetime import datetime

from keras import Sequential
import numpy as np
from src.pipeline.stage import Stage
from src.stages.execution.plotter import Plotter


class Builder(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)

        self.batch_size = self.config["network"]["batch"]
        self.epochs = self.config["network"]["epochs"]
        self.log_level = self.config["network"]["log"]

        self.date_time_formatted = datetime.now().strftime("%d-%m-%Y-%H:%M")
        self.model_name = None

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
        self.model_name = f"lstm_{self.get_attributes('crypto')}_{self.date_time_formatted}"
        results = self.model.fit(self.get_attributes('train_x'),
                                 self.get_attributes('train_y'),
                                 validation_data=(self.get_attributes('test_x'), self.get_attributes('test_y')),
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=self.log_level, shuffle=False)

        if not os.path.exists("../results/graphs"):
            os.makedirs("../results/graphs")

        # plotting loss
        history = results.history
        self.plotter.plot(data=[history['val_loss'], history['loss']],
                          legend=['val_loss', 'loss'],
                          title='Loss',
                          x_label='Epochs',
                          y_label='Loss',
                          save_name=f"../results/graphs/loss_{self.model_name}")

        # plotting metric
        self.plotter.plot(data=[history[f'val_{self.model.metrics_names[1]}'], history[self.model.metrics_names[1]]],
                          legend=[f'val_{self.model.metrics_names[1]}', self.model.metrics_names[1]],
                          title=self.model.metrics_names[1],
                          x_label='Epochs',
                          y_label='Metric',
                          save_name=f"../results/graphs/metric_{self.model_name}")

    """
    Verifying model
    """

    def verify(self):
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
        print("Sum of differences between predictions: ", self.calculate_diff_sum(prediction, actual))

        self.plotter.plot(data=[prediction, actual],
                          legend=['Predicted', 'True'],
                          title="Closing Prices",
                          x_label="Day",
                          x_ticks=1.0,
                          y_label="Price",
                          save_name=f"../results/graphs/plot_{self.model_name}")

    """
    Calculating sum of differences between real and predicted prices
    """

    def calculate_diff_sum(self, prediction: list, real: list):
        diff = 0

        for index, real_price in enumerate(real):
            diff += abs(real_price - prediction[index])

        return diff

    """
    Saving trained model with weights.
    """

    def save(self):
        if not os.path.exists("../results/model"):
            os.makedirs("../results/model")

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(f"../results/model/{self.model_name}.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        with open(f'../results/model/{self.model_name}_weights.h5', "wb") as file:
            self.model.save_weights(file)

    def run(self):
        self.build()
        self.train()
        self.verify()
        self.save()

    def test(self):
        pass
