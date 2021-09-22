from datetime import datetime
from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *


def main():
    start = datetime(2020, 1, 1, 0).strftime('%m/%d/%Y %H:%M')
    end = datetime(2020, 5, 31, 0).strftime('%m/%d/%Y %H:%M')
    s1 = Sensor.getSensor(11563)
    print(s1)
    measures = s1.getMeasures(start_interval=start, end_interval=end)
    my_agent = MovingAverageV2()
    my_agent.makePrediction(measures)
    my_agent_2 = MovingAverageV3()
    my_agent_2.makePrediction(measures)
    #viable options ATM: pm1, pm10, pm25, temp

    # print(s1.nearestNeighbors(2))


if __name__ == '__main__':
    main()
