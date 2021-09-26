from datetime import datetime
from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *
from edu.pwr.map.Model import *


def main():
    #modelSummary(getJson("C:\\Users\\User\\PycharmProjects\\AirBots\\Docs\\Model1"))

    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 7, 0)
    sensor = getSensorORM(11563)
    data = sensor.getMeasures(start, end)
    agent = MovingAverageV1()
    print(agent.cf)
    agent.makePrediction(data)
    print(agent.cf)


if __name__ == '__main__':
    main()
