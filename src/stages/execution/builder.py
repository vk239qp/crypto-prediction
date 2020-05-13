import os
import statistics
from abc import abstractmethod
from datetime import datetime

from keras import Sequential
import numpy as np
from src.pipeline.stage import Stage
from src.stages.execution.plotter import Plotter

"""
Calculating the median
first we calculate percentage deviations for every day, after that return the median
"""


def get_median_of_deviations(prediction, actual):
    if len(prediction) < 1 and len(actual) < 1:
        return 0
    diff = list()
    for act, pred in zip(actual, prediction):
        actual_price, pred_price = act[0], pred[0]
        diff.append(((abs(actual_price - pred_price) / actual_price) * 100))

    return statistics.median(diff)


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

        print(f"Avarage of loss: {sum(results.history['loss']) / len(results.history['loss'])}")
        print(f"Avarage of rmse: {sum(results.history['rmse']) / len(results.history['rmse'])}")

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
        print("Median of deviations:\n", get_median_of_deviations(prediction, actual))

        self.plotter.plot(data=[prediction, actual],
                          legend=['Predicted', 'True'],
                          title="Closing Prices",
                          x_label="Day",
                          x_ticks=1.0,
                          y_label="Price",
                          save_name=f"../results/graphs/plot_{self.model_name}")

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
