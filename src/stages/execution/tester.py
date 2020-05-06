import os

import numpy as np
import pandas as pd
from keras.engine.saving import model_from_json

from src.pipeline.stage import Stage
from src.stages.execution.plotter import Plotter


class Tester(Stage):

    def __init__(self, config_file: str, only_compare=False):
        super().__init__(config_file)
        self.only_compare = only_compare
        self.model_file = self.config["model"]["name"]
        self.days_to_compare = self.config["model"]["days_to_compare"]

        # load json and create model
        with open(f"../results/model/{self.model_file}.json", 'r') as model:
            self.model = model_from_json(model.read())

        # load weights into new model
        self.model.load_weights(f"../results/model/{self.model_file}_weights.h5")
        self.plotter = Plotter()

    """
    Verify prediction of the model by comparing them with data from last N days.
    """

    def compare(self):
        # loading newest data from N days
        data_real = pd.read_csv(f'../dataset/prices_{self.get_attributes("crypto")}.csv').tail(
            self.days_to_compare).reset_index(drop=True)
        data_real = data_real["close"]

        # loading predicted data
        data_prediction = np.loadtxt(f"../results/predictions/prediction_data_{self.model_file}.txt")

        self.plotter.plot(data=[data_prediction, data_real],
                          legend=['Predicted', 'Real'],
                          title="Real vs Predicted prices",
                          x_label="Day",
                          x_ticks=1.0,
                          time=False,
                          y_label="Price",
                          save_name=f"../results/predictions/compare_{self.model_file}")

    def predict(self):
        scaler = self.get_attributes('scaler_y')
        test_x = self.get_attributes('test_x')
        past_steps = self.get_attributes('past_steps')
        features = self.get_attributes('features')

        prediction = self.model.predict(test_x[-1].reshape(1, past_steps, features)).tolist()[0]
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()

        if not os.path.exists("../results/predictions"):
            os.makedirs("../results/predictions")

        with open(f"../results/predictions/prediction_data_{self.model_file}.txt", 'w') as file:
            for price in prediction:
                file.write(f"{price}\n")

        self.plotter.plot(data=[prediction],
                          legend=['Predicted'],
                          title=f"Model {self.model_file} prediction",
                          x_label="Day",
                          x_ticks=1.0,
                          time=False,
                          save_name=f"../results/predictions/prediction_{self.model_file}",
                          y_label="Closing price")

    def run(self):
        pass

    def test(self):
        if self.only_compare:
            self.compare()
        else:
            self.predict()
