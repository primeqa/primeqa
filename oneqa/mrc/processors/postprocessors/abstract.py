import logging
from abc import ABCMeta, abstractmethod


class AbstractPostProcessor(metaclass=ABCMeta):
    def __init__(self, k: int):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._k = k

    @abstractmethod
    def process(self, examples, features, predictions):
        pass  # TODO types on args?
