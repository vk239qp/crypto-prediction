import matplotlib.pyplot as plt


class Plotter:
    """
    Showing graph.

    data - Array of data to be plotted.
    legend - Legend of the graph.
    title - Title of the graph.
    x_label - X axis label.
    y_label - Y axis label.
    """

    def show(self, data: [], legend: [], title: str, x_label: str, y_label: str):
        plt.figure(figsize=(12, 4))

        for data_sample in data:
            plt.plot(data_sample)

        plt.legend(legend)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
