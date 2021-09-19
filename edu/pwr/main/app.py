from datetime import datetime
from edu.pwr.map.Models import *


def main():
    start = datetime(2020, 1, 1, 0).strftime('%m/%d/%Y %H:%M')
    end = datetime(2020, 1, 2, 0).strftime('%m/%d/%Y %H:%M')
    s1 = Sensor.getSensor(11563)
    print(s1)
    results = s1.getMeasures(start_interval=start, end_interval=end)
    #viable options ATM: pm1, pm10, pm25, temp
    prepareMeasures(results, "pm1")
    # print(s1.nearestNeighbors(2))


if __name__ == '__main__':
    main()
