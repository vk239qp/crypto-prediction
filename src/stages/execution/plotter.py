from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Showing and saving graph.

    data - Array of data to be plotted.
    legend - Legend of the graph.
    title - Title of the graph.
    x_label - X axis label.
    y_label - Y axis label.
    save_name - Path of the file where graph will be stored.
    time - Boolean whether to add timestamp to graph name.
    x_ticks - Setting x-axis range.
    """

    def plot(self, data: [], legend: [], title: str, x_label: str, y_label: str, save_name: str, x_ticks: float = None,
             time=True):
        date_time = datetime.now()
        date_time_formatted = date_time.strftime("%d-%m-%Y-%H:%M")

        plt.figure(figsize=(12, 4))

        for data_sample in data:
            plt.plot(data_sample)

        if x_ticks is not None:
            plt.xticks(np.arange(0, len(data[0]), 1))

        plt.legend(legend)

        if time:
            plt.title(f"{title} {date_time_formatted}")
        else:
            plt.title(title)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        plt.savefig(f"{save_name}.png")
