from datetime import datetime
from src.map.Central import *


def sensorCheck(start, end):
    sens = getSensorsORM()
    for s in sens:
        data = getMeasuresORM(s.sid, start, end)
        print(s.sid)
        if data:
            print(len(data)/countInterval(start, end))
            for d in data:
                print(d)
        else:
            print("no data")


def main():
    # archiveResults('..\\..\\..\\AirBots\\docs')
    # sensorCheck(datetime(2020, 12, 14, 12), datetime(2020, 12, 15, 0))
    # model = Central('..\\..\\..\\AirBots\\docs\\Model2')
    # model.makePrediction(13128, datetime(2020, 3, 15, 0))
    model = Central('..\\..\\..\\AirBots\\docs\\Model2')
    model.makePrediction(5697, datetime(2020, 4, 15, 0), "pm1")
    

if __name__ == '__main__':
    main()
