from src.pipeline.stage import Stage
from keras.models import load_model
import numpy as np

from src.stages.execution.plotter import Plotter


class Tester(Stage):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.model_file = self.config["model"]["name"]
        self.model = load_model(f"../results/model/{self.model_file}.h5")
        self.plotter = Plotter()

    def predict(self):
        scaler = self.get_attributes('scaler_y')
        test_x = self.get_attributes('test_x')
        past_steps = self.get_attributes('past_steps')
        features = self.get_attributes('features')

        prediction = self.model.predict(test_x[-1].reshape(1, past_steps, features)).tolist()[0]
        prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))

        self.plotter.show(data=[prediction],
                          legend=['Predicted'],
                          title=f"Model {self.model_file} prediction",
                          x_label="Day",
                          x_ticks=1.0,
                          time=False,
                          y_label="Closing price")

        self.plotter.save(f"prediction_{self.get_attributes('crypto')}_{self.model_file}.png")

    def run(self):
        pass

    def test(self):
        self.predict()
