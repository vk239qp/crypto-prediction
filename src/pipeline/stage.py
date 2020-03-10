from abc import ABC, abstractmethod

"""
Pipeline component class
"""


class Stage(ABC):
    """
    This method must be implemented in concrete pipeline component. It defines component's logic.
    """

    @abstractmethod
    def run(self):
        pass
