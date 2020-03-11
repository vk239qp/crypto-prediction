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

        self.pipe = None

    """
    This method must be implemented in concrete pipeline component. It defines component's logic.
    """

    @abstractmethod
    def run(self):
        pass

    """
    Obtaining access to pipe and all attributes pushed to the flow.
    """

    def attach(self, pipe):
        self.pipe = pipe

    """
    This method pushes component's attributes to the flow.
    """

    def push_attributes(self):
        return vars(self)

    """
    This method gets attribute from the flow.
    """

    def get_attributes(self, property_name: str):
        return self.pipe.properties[property_name]
