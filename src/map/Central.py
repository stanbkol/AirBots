from datetime import datetime, timedelta
import json

import numpy as np

from src.database.Models import *
from src.map.TileClassifier import *
from src.agents.Agents import *
from src.database.DataLoader import *
from sklearn.metrics import mean_squared_error


def findNearestSensors(sensorid, s_list, n):
    """

    :param sensorid: sensor for which to find nearest neighbors.
    :param s_list: list of active sensors
    :return: list of (Sensor object, meters distance) tuples
    """
    base_sensor = getSensorORM(sensorid)
    sensors_orm = []

    for s in s_list:
        if s != sensorid:
            sensors_orm.append(getSensorORM(s))

    distances = []
    startLL = MapPoint(base_sensor.lat, base_sensor.lon)
    for sensor in sensors_orm:
        meters_away = calcDistance(startLL, MapPoint(sensor.lat, sensor.lon))
        distances.append((sensor, meters_away))

    distances.sort(key=lambda x: x[1])
    return distances[:n]


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
    summary = {"sensor": sid, "first": earliest_date.strftime("%D %H"), "last": latest_date.strftime("%D %H"),
               "num_intervals": observed,
               "total": countInterval(datetime.datetime(2018, 9, 3, 0), datetime.datetime(2021, 5, 5, 0))}
    return summary


def countInterval(start, end):
    diff = end - start
    days, seconds = diff.days, diff.seconds
    total_intervals = days * 24 + seconds // 3600
    return total_intervals


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


def MSE(a, p):
    actual, pred = np.array(a), np.array(p)
    return mean_squared_error(actual, pred)


def aggregatePrediction(preds):
    print(preds)
    print(len(preds))
    total = 0
    for k in preds:
        total += preds[k]
    return round(total/(len(preds)), 2)


def makeCluster(sid, sensors, n):
    data = findNearestSensors(sid, sensors, n)
    cluster = []
    for sens, dist in data:
        cluster.append(sens.sid)
    return cluster


class Central:
    agents = {}
    sensors = []
    agent_configs = {}
    thresholds = {}

    def __init__(self, model):
        self.model_file = model
        self.data = getJson(self.model_file)
        self.model_params = self.data["model_params"]
        self.error = 0
        self.popSensors()
        self.extractData()

    def evaluateAgents(self, values, predictions):
        print("values:")
        print(values)
        for a in self.agents:
            print("predictions for:", a.sid)
            print(predictions[a.sid])
            a.error = MSE(values, predictions[a.sid])

    def popSensors(self):
        for s in self.data["sensors"]["on"]:
            self.sensors.append(s)

    def getAllPredictions(self, target, time, val):
        predictions = {}
        for a in self.agents:
            agent = self.agents[a]
            predictions[a] = agent.makePredictions(target, time, ["pm1"], meas=val)
        return predictions

    def trainModel(self):
        start_interval = datetime.strptime(self.model_params["start_interval"], '%Y-%m-%d %H:%M')
        end_interval = datetime.strptime(self.model_params["end_interval"], '%Y-%m-%d %H:%M')
        target = self.model_params["target"]
        for i in range(1, self.model_params["num_iter"] + 1):
            cursor = start_interval
            predictions = {sid:[] for sid in self.sensors}
            values = []
            model_vals = []
            while cursor != end_interval:
                print("Predictions for ", cursor)
                val = getMeasureORM(target, cursor)
                vals = {sid:[] for sid in self.sensors}
                if val:
                    values.append(val.pm1)

                    interval_preds = self.getAllPredictions(target, cursor, val)
                    print(interval_preds)
                    #for each agent, call collabPrediction
                    for a in self.agents:
                        self_pred = interval_preds[a]
                        cluster_pred = {}
                        agent = self.agents[a]
                        for ca in agent.cluster:
                            cluster_pred[ca] = interval_preds[ca]
                        pred = (self_pred, cluster_pred)
                        print("Agent: ", a)
                        print("Prediction: ", pred)
                        #pred = a.collabPrediction(self_pred, cluster_pred)
                        # vals[a.sid] = (round(pred[0], 2))
                        # predictions[a.sid].append(round(pred[0], 2))
                else:
                    print("No target validation data for-->", cursor)
                # model_vals.append(aggregatePrediction(vals))
                cursor += timedelta(hours=1)
            # self.evaluateAgents(values, predictions)
            # self.error = MSE(values, model_vals)
            # print("Model MSE=", self.error)
            # self.saveModel(i)

    def sensorSummary(self):
        for s in self.sensors:
            print(examineSensor(s))

    def extractData(self):
        self.thresholds = self.data["thresholds"]
        self.agent_configs = self.data["agent_configs"]
        for s in self.sensors:
            cluster = makeCluster(s, self.sensors, self.model_params['cluster_size'])
            a = Agent(s, self.thresholds, cluster, config=self.agent_configs)
            self.agents[a.sid] = a

    def makePrediction(self, target, time):
        predictions = []
        for a in self.agents:
            print(type(a))
            p = a.makePrediction(target, time)
            print(p)
            predictions.append(p)
        return predictions

    def saveModel(self, i):
        with open(self.model_file + "_results", 'w+', encoding="utf-8") as f:
            f.write("Iteration #" + str(i) + "\n")
            for a in self.agents:
                f.write("SID:" + str(a.sid) + "\n")
                f.write("MSE=" + str(a.error) + "\n")
            f.write("Model Aggregation" + "\n")
            f.write("MSE=" + str(self.error) + "\n")
        f.close()
