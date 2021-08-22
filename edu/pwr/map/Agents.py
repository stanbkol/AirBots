from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def makePrediction(self, data):
        pass


class MovingAverageV1(Agent):

    def makePrediction(self, data):
        pass


class MultiDimensionV1(Agent):

    def makePrediction(self, data):
        pass
