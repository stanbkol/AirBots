from datetime import datetime
from src.agents.ForecastModels import MultiVariate


def testModel():
    sensors = [
        11553,
        11563,
        11571,
        11583,
        11585,
        11587,
        11596,
        11597,
        11619,
        11640
    ]
    thresholds = {}
    marvin = MultiVariate(11563, sensors, thresholds)
    date = datetime.datetime(year=2019, month=1, day=6, hour=0)
    data, cols = marvin._prepareData(11563, date, day_interval=2, hour_interval=0, targetObs=['pm1'])
    for d in data:
        print(d)

if __name__ == '__main__':
    pass