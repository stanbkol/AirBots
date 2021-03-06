from datetime import datetime

from src.database.Models import getMeasuresORM, getSensorsORM
from src.main.utils import countInterval
from src.map.Central import Central, archiveResults


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


def run():
    # archiveResults('..\\..\\..\\AirBots\\docs')
    model = Central('..\\..\\..\\AirBots\\docs\\Model2')
    model.makePrediction(5697, datetime(2020, 2, 15, 0), "pm10")


if __name__ == '__main__':
    run()
