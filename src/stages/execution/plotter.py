import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self):
        self.plot = plt.figure(figsize=(12, 4))

    """
    Showing graph.

    data - Array of data to be plotted.
    legend - Legend of the graph.
    title - Title of the graph.
    x_label - X axis label.
    y_label - Y axis label.
    time - Boolean whether to add timestamp to graph name.
    x_ticks - Setting x-axis range.
    """

    def show(self, data: [], legend: [], title: str, x_label: str, y_label: str, x_ticks: float = None, time=True):
        date_time = datetime.now()
        date_time_formatted = date_time.strftime("%d-%m-%Y-%H:%M")

        for data_sample in data:
            self.plot.plot(data_sample)

        if x_ticks is not None:
            self.plot.xticks(np.arange(1, len(data[0] + 1), 1))

        self.plot.legend(legend)

        if time:
            self.plot.title(f"{title} {date_time_formatted}")
        else:
            self.plot.title(title)

        self.plot.xlabel(x_label)
        self.plot.ylabel(y_label)
        self.plot.show()

    """
    Saving plot.
    
    file_name - name of the file to save.
    """

    def save(self, file_name: str):
        if not os.path.exists("../results/graphs/predictions"):
            os.makedirs("../results/graphs/predictions")

        plt.savefig(f"../results/graphs/predictions/{file_name}")
