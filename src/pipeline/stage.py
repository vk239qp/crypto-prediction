from abc import ABC, abstractmethod

import yaml

"""
Pipeline component class
"""


class Stage(ABC):
    """
    config_file - Name of the config file for specific stage.
    """

    def __init__(self, config_file: str):
        with open(f"../src/config/{config_file}.yml", 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

    """
    This method must be implemented in concrete pipeline component. It defines component's logic.
    """

    @abstractmethod
    def run(self):
        pass

    """
    This method gathers all component's attributes.
    """

    def get_attributes(self):
        return vars(self)
