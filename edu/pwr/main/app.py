from datetime import datetime
from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *


def main():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 2, 0)
    s1 = Sensor.getSensor(11563)
    measures = s1.getMeasures(start_interval=start, end_interval=end)
    my_agent_2 = MultiDimensionV1()
    #my_agent_2.makePrediction(measures)
    # print(s1.nearestNeighbors(2))


if __name__ == '__main__':
    main()
