from edu.pwr.map.Models import *
from edu.pwr.map.Agents import *
from dateutil import parser
import json


def makeModel(file):
    getJson(file)


def getJson(file):
    f = open(file, )
    data = json.load(f)
    return data


def countInterval(start, end):
    diff = end-start
    days, seconds = diff.days, diff.seconds
    total_intervals = days * 24 + seconds // 3600
    return total_intervals+1


def rateInterval(dataset, total):
    return dataset/total


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