from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """
    Showing graph.

    data - Array of data to be plotted.
    legend - Legend of the graph.
    title - Title of the graph.
    x_label - X axis label.
    y_label - Y axis label.
    """

    def show(self, data: [], legend: [], title: str, x_label: str, y_label: str, x_ticks: float = None):
        date_time = datetime.now()
        date_time_formatted = date_time.strftime("%d-%m-%Y-%H:%M")

        plt.figure(figsize=(12, 4))

        for data_sample in data:
            plt.plot(data_sample)

        if x_ticks is not None:
            plt.xticks(np.arange(1, len(data[0] + 1), 1))

        plt.legend(legend)
        plt.title(f"{title} {date_time_formatted}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
