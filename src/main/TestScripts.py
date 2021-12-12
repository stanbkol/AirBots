from datetime import datetime
from src.map.Central import *


def iterationTests(model):
    model.iterations = 5
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "iteratons")
    model.iterations = 10
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "iteratons")
    model.iterations = 15
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "iteratons")


def intervalTests(model):
    model.interval = 6
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "intervals")
    model.interval = 12
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "intervals")
    model.interval = 24
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "intervals")


def clusterTests(model):
    model.cluster_size = 3
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "cluster")
    model.cluster_size = 5
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "cluster")
    model.cluster_size = 7
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "cluster")


def ratioTests(model):
    model.ratio = 0.6
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "ratio")
    model.ratio = 0.7
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "ratio")
    model.ratio = 0.8
    model.makePrediction(5697, datetime(2020, 3, 15, 0), "ratio")


def run():
    model = Central('..\\..\\..\\AirBots\\docs\\Model2')
    iterationTests(model)
    model.iterations = 5
    intervalTests(model)
    model.interval = 12
    clusterTests(model)
    model.cluster_size = 3
    ratioTests(model)


if __name__ == '__main__':
    run()