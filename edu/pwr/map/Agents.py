from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def makePrediction(self):
        pass


class MovingAverageV1(Agent):

    def makePrediction(self):
        pass


class MultiDimensionV1(Agent):

    def makePrediction(self):
        pass
