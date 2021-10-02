from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *
from dateutil import parser
import json
import pandas


def makeModel(file):
    getJson(file)


# returns a dictionary containing the json file
def getJson(file):
    f = open(file, )
    data = json.load(f)
    return data


def countInterval(start, end):
    diff = end-start
    days, seconds = diff.days, diff.seconds
    total_intervals = days * 24 + seconds // 3600
    return total_intervals


def fillInterval(startMeasure, endMeasure):
    avg_pm1 = (startMeasure.pm1 + endMeasure.pm1) / 2
    avg_pm10 = (startMeasure.pm10 + endMeasure.pm10) / 2
    avg_pm25 = (startMeasure.pm25 + endMeasure.pm25) / 2
    avg_temp = (startMeasure.temp + endMeasure.temp) / 2
    time_range = pandas.date_range(startMeasure.date, endMeasure.date, freq='H')
    Soft_Data = []
    for e in time_range:
        dk = int(e.strftime('%Y%m%d%H'))
        Soft_Data.append(
            Measure(date_key=dk, sensor_id=startMeasure.sid, date=e, pm1=avg_pm1, pm10=avg_pm10, pm25=avg_pm25,
                    temperature=avg_temp))
    Soft_Data.pop(0)
    Soft_Data.pop()
    return Soft_Data


def cleanInterval(data):
    for i, x in zip(data, data[1:]):
        if (countInterval(i.date, x.date)) != 1:
            data.extend(fillInterval(i, x))
    return sorted(data, key=lambda x: x.dk)


def rateInterval(dataset, total):
    return (dataset-1)/total


def modelSummary(model_data):
    start = parser.parse(model_data["start_date"])
    end = parser.parse(model_data["end_date"])
    total = countInterval(start, end)
    threshold = model_data["accuracy_threshold"]
    print("Sensor Summary from file..")
    print("Total possible measures in interval= ", countInterval(start, end))
    sensor_data = model_data["sensors"]["on"]
    sensors_passed = []
    sensors_failed = []
    print(threshold)
    for s in sensor_data:
        sensor = getSensorORM(s)
        data = sensor.getMeasures(start, end)
        completeness = rateInterval(len(data), total)
        if completeness > threshold:
            sensors_passed.append((sensor, completeness))
        else:
            sensors_failed.append((sensor, completeness))
    print("Sensors Passed=", len(sensors_passed))
    print("---------------------------------------------------------")
    for s in sensors_passed:
        print(s[0])
        print("Completeness-->%", s[1]*100)
    print("Sensors Failed=", len(sensors_failed))
    print("---------------------------------------------------------")
    for s in sensors_failed:
        print(s)