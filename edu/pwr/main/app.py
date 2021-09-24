from datetime import datetime
from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *


def main():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 5, 31, 0)
    s1 = Sensor.getSensor(11563)
    print(s1)
    measures = s1.getMeasures(start_interval=start, end_interval=end)
    # my_agent = MovingAverageV2()
    # my_agent.makePrediction(measures, 10)
    my_agent_2 = MovingAverageV1()
    my_agent_2.makePrediction(measures, 10)
    # print(s1.nearestNeighbors(2))


if __name__ == '__main__':
    main()
