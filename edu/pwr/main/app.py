from datetime import datetime
from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *
from edu.pwr.map.Model import *


def main():
    # modelSummary(getJson("C:\\Users\\User\\PycharmProjects\\AirBots\\Docs\\Model1"))
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 7, 0)
    sensor = getSensorORM(11640)
    data = sensor.getMeasures(start, end)
    sorted_data = sorted(data, key=lambda x: x.dk)
    final_data = cleanInterval(sorted_data)
    for m in final_data:
        print(m)


if __name__ == '__main__':
    main()
