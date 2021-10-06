from dateutil import parser
import json
import pandas

from src.database.Models import getSensorORM, Measure


class Model:

    def __init__(self, file):
        self.model_file = file
        self.data = getJson(file)
        self.start = parser.parse(self.data["start_date"])
        self.end = parser.parse(self.data["end_date"])
        self.threshold = self.data["accuracy_threshold"]
        self.target = getSensorORM(self.data["sensors"]["target"])
        self.sensors = self.populateSensors()
        self.sensor_data = {}
        self.cleaned_data = {}
        self.filterModel()
        self.cleanModel()
        self.runModel()

    def populateSensors(self):
        s_list = fetchSensors(self.data["sensors"]["on"])
        s_dict = {}
        for s in s_list:
            s_dict[s.sid] = s
        return s_dict

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
            #print("Sensor-->", s)
            temp = self.sensor_data[s]
            if len(temp) != total:
                #print("Cleaning interval..")
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
                #print("Removing Sensor: "+str(k))
                print(sensors_copy.pop(k))
        print(len(sensors_passed))
        self.sensors = sensors_copy
        self.sensor_data = sensors_passed

    # once data has been confirmed to be clean, begin iterations to make predictions on provided data
    def runModel(self):
        self.saveModel(self.model_file+"_Results")

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
                f.write("Data for sensor #"+str(s) + '\n')
                f.write("Provided Entries: "+str(len(self.sensor_data[s])) + '\n')
                # add if statement block to check if interval was complete, else print amount of cleaned entries
                #f.write("Entries After Clean: "+str(len(self.cleaned_data[s])))
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
    Soft_Data.pop(0)
    Soft_Data.pop()
    return Soft_Data


def cleanInterval(data):
    for first, second in zip(data, data[1:]):
        if (countInterval(first.date, second.date)) != 1:
            data.extend(fillInterval(first, second))
    return sorted(data, key=lambda x: x.dk)


def rateInterval(dataset, total):
    return (dataset - 1) / total


def fetchSensors(sensor_list):
    sensors = []
    for s in sensor_list:
        sensors.append(getSensorORM(s))
    return sensors


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
