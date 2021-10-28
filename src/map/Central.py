from datetime import datetime

from dateutil import parser
import json
import pandas

from src.database.Models import getSensorORM, Measure, getTileORM
from src.map.MapPoint import MapPoint, calcDistance
from src.map.TileClassifier import *
from src.agents.Agents import *


def generateAgent(sid, name):
    agent_options = {
        "random": randomAgent(sensor_id=sid),
        "simple_avg": simpleAgentV1(sensor_id=sid),
        "minmax_avg": simpleAgentV2(sensor_id=sid),
        "nearest": simpleAgentV3(sensor_id=sid),
        "WMA": MovingAverageV1(sensor_id=sid),
        "SMA": MovingAverageV2(sensor_id=sid),
        "ARIMA": ARMIAX(sensor_id=sid),
        "MVR": MultiDimensionV1(sensor_id=sid)
    }
    a = agent_options.get(name)
    return a


class Central:
    agents = []
    sensors = []
    agent_configs = {}
    thresholds = {}

    def __init__(self, file):
        self.model_file = file
        self.data = getJson(file)
        self.extractData()
        self.sensorSummary()

    def sensorSummary(self):
        for s in self.sensors:
            print(examineSensor(s[0]))

    def extractData(self):
        self.thresholds = self.data["thresholds"]
        self.sensors = self.data["sensors"]["on"]
        sensor_list = [x[0] for x in self.sensors]
        self.agent_configs = self.data["agent_configs"]
        for s in self.sensors:
            a = generateAgent(s[0], s[1])
            a.configs = self.agent_configs[s[1]]
            a.sensor_list = sensor_list
            self.agents.append(a)

    def makePrediction(self, target, time):
        predictions = []
        for a in self.agents:
            print(type(a))
            p = a.makePrediction(target, time)
            print(p)
            predictions.append(p)
        return predictions

    def printModel(self):
        print("Model Summary")
        print("Start: ", self.start)
        print("End: ", self.end)
        print("Completeness Threshold: ", self.threshold)
        print("Total Sensors: ", len(self.data["sensors"]["on"]))
        print("Final Sensors: ", len(self.sensor_data))

    # iterate through all sensors in sensor_data, clean each interval and update the sensor_data list
    def cleanModel(self):
        print("Cleaning the data..")
        total = countInterval(self.start, self.end)
        for s in self.sensor_data:
            # print("Sensor-->", s)
            temp = self.sensor_data[s]
            if len(temp) != total:
                # print("Cleaning interval..")
                self.cleaned_data[s] = cleanInterval(temp)
            # else:
            #     print("Interval Complete..")

    def filterModel(self):
        sensors_copy = self.sensors.copy()
        sensors_passed = {}
        print(len(sensors_copy))
        for k in self.sensors:
            entries = self.sensors[k].getMeasures(self.start, self.end)
            if (len(entries) / countInterval(self.start, self.end)) > self.threshold:
                sensors_passed[k] = sorted(entries, key=lambda x: x.dk)
            else:
                # print("Removing Sensor: "+str(k))
                print(sensors_copy.pop(k))
        print(len(sensors_passed))
        self.sensors = sensors_copy
        self.sensor_data = sensors_passed

    # once data has been confirmed to be clean, begin iterations to make predictions on provided data
    def runModel(self, target, time):
        self.saveModel(self.model_file + "_Results")

    def saveModel(self, filename):
        with open(filename, 'w', encoding="utf-8") as f:
            f.write("Model Summary for: " + self.model_file + '\n')
            f.write("Start of Interval: " + str(self.start) + '\n')
            f.write("End of Interval: " + str(self.end) + '\n')
            f.write("Accuracy Threshold: " + str(self.threshold) + '\n')
            f.write("Confidence Factor Threshold: " + '\n')
            f.write("Target Sensor: " + str(self.target.sid) + '\n')
            f.write("The Chosen Sensors..\n")
            print(len(self.sensors))
            for s in self.sensors:
                f.write("Data for sensor #" + str(s) + '\n')
                f.write("Provided Entries: " + str(len(self.sensor_data[s])) + '\n')
                # add if statement block to check if interval was complete, else print amount of cleaned entries
                # f.write("Entries After Clean: "+str(len(self.cleaned_data[s])))
        f.close()


def fetchBB(data):
    bounding_box = []
    for entry in data:
        bounding_box.append(entry["boundingbox"])
    return bounding_box[0]


# returns a dictionary containing the json file
def getJson(file):
    f = open(file, encoding="utf8")
    data = json.load(f)
    return data

# check database for all measurements data for given sensor, using different granularity (ie. for full dataset,
# by year, by month, by week, by day); should it be done with multiple, specific queries? or taking the full dataset
# for a sensor and breaking it down using additional algorithms
def examineSensor(sid, g="*"):
    full_dataset = getMeasuresORM(sid)
    earliest_date = full_dataset[0].date
    latest_date = full_dataset[-1].date
    total = countInterval(earliest_date, latest_date)
    observed = len(full_dataset)
    summary = {"sensor": sid, "first": earliest_date, "last": latest_date, "num_intervals": observed, "completion_v1": observed / total, "completion_v2": observed / countInterval(datetime(2017, 11, 12, 0), datetime(2021, 5, 5, 0))}
    return summary

# utilize threshold passed into json file, ie. sensor must have data within 48 intervals of chosen time, otherwise it
# is not worth cleaning data for prediction
def examineSensorDate(sid, date):
    pass


def countInterval(start, end):
    diff = end - start
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
    return Soft_Data


def cleanInterval(data):
    for first, second in zip(data, data[1:]):
        if (countInterval(first.date, second.date)) != 1:
            filled = fillInterval(first, second)
            filled.pop(0)
            filled.pop()
            data.extend(filled)
    return sorted(data, key=lambda x: x.dk)


def rateInterval(dataset, total):
    return (dataset - 1) / total


def fetchSensors(sensor_list):
    sensors = []
    for s in sensor_list:
        sensors.append(getSensorORM(s))
    return sensors


def classifyTiles(filename):
    data = getJson(filename)
    for tile in data:
        print("Classifying Tile:", tile)
        tile_class = classifyT(data[tile])
        print("Class-->", tile_class)
        tile_obj = getTileORM(tile)
        tile_obj.setClass(tile_class)


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
        print("Completeness-->%", s[1] * 100)
    print("Sensors Failed=", len(sensors_failed))
    print("---------------------------------------------------------")
    for s in sensors_failed:
        print(s)
