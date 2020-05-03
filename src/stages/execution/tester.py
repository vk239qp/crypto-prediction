import os

import numpy as np
from keras.engine.saving import model_from_json

from src.pipeline.stage import Stage
from src.stages.execution.plotter import Plotter


class Tester(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.model_file = self.config["model"]["name"]
        # load json and create model
        with open(f"../results/model/{self.model_file}.json", 'r') as model:
            self.model = model_from_json(model.read())

        # load weights into new model
        self.model.load_weights(f"../results/model/{self.model_file}_weights.h5")
        self.plotter = Plotter()

    def predict(self):
        scaler = self.get_attributes('scaler_y')
        test_x = self.get_attributes('test_x')
        past_steps = self.get_attributes('past_steps')
        features = self.get_attributes('features')

        prediction = self.model.predict(test_x[-1].reshape(1, past_steps, features)).tolist()[0]
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

        if not os.path.exists("../results/graphs/predictions"):
            os.makedirs("../results/graphs/predictions")

        self.plotter.plot(data=[prediction],
                          legend=['Predicted'],
                          title=f"Model {self.model_file} prediction",
                          x_label="Day",
                          x_ticks=1.0,
                          time=False,
                          save_name=f"../results/graphs/predictions/prediction_{self.model_file}",
                          y_label="Closing price")

    def run(self):
        pass

    def test(self):
        self.predict()
