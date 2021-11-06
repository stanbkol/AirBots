from datetime import datetime
from src.database.Models import sensorMerge, populateTables
from src.map.Central import *


def main():
    c = Central('..\\..\\..\\AirBot\\docs\\Model1')
    # data = getMeasuresORM(11571, datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 30, 0))
    # for e in data:
    #     print(e)
    # print(len(data))
    # c.sensorSummary()


if __name__ == '__main__':
    main()
