import datetime
import json
from src.database.Models import *
from src.map.TileClassifier import *
from src.agents.Agents import *
from src.database.DataLoader import *


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
    summary = {"sensor": sid, "first": earliest_date, "last": latest_date, "num_intervals": observed,
               "completion_v1": observed / total,
               "completion_v2": observed / countInterval(datetime(2017, 11, 12, 0), datetime(2021, 5, 5, 0))}
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


def MSE(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual, pred)).mean()


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

    def __init__(self, model):
        self.model_file = model
        self.data = getJson(self.model_file)
        self.model_params = self.data["model_params"]
        self.extractData()
        self.trainModel()

    def trainModel(self):
        start_interval = datetime.datetime.strptime(self.model_params["start_interval"], '%Y-%m-%d %H:%M')
        end_interval = datetime.datetime.strptime(self.model_params["end_interval"], '%Y-%m-%d %H:%M')
        for i in range(1, self.model_params["num_iter"] + 1):
            for a in self.agents:
                cursor = start_interval
                print(a["type"])
                predictions = []
                values = []
                while cursor != end_interval:
                    # try:
                    pred = a["agent"].makePrediction(6494, start_interval)
                    val = getMeasureORM(6494, start_interval).pm1
                    predictions.append(pred)
                    values.append(val)
                    # except:
                    #     print("Error, SID="+str(a["sensor"])+" at" + str(start_interval))
                    # if MSE < 0.3:
                    #     a["trust"] = a["trust"] + 1
                    # else:
                    #     a["trust"] = a["trust"] - 1
                    cursor += datetime.timedelta(hours=1)
                a["trust"] = MSE(predictions, values)
            self.saveModel(i)

    def sensorSummary(self):
        for s in self.sensors:
            print(examineSensor(s[0]))

    def extractData(self):
        self.thresholds = self.data["thresholds"]
        self.sensors = self.data["sensors"]["on"]
        self.agent_configs = self.data["agent_configs"]
        for s in self.sensors:
            a = generateAgent(s[0], s[1])
            a.configs = self.agent_configs[s[1]]
            agent_dict = {
                "sensor": a.sensor.sid,
                "agent": a,
                "type": s[1],
                "trust": 0,
                "configs": a.configs
            }
            self.agents.append(agent_dict)

    def makePrediction(self, target, time):
        predictions = []
        for a in self.agents:
            print(type(a))
            p = a.makePrediction(target, time)
            print(p)
            predictions.append(p)
        return predictions

    def saveModel(self, i):
        with open(self.model_file + "_results", 'w', encoding="utf-8") as f:
            f.write("Iteration #" + str(i) + "\n")
            for a in self.agents:
                f.write("SID:" + str(a["sensor"]) + "\n")
                f.write("Agent:" + a["type"] + "\n")
                f.write("TF=" + str(a["trust"]) + "\n")
                f.write("Agent Configs" + "\n")
                f.write(json.dumps(a["configs"]) + "\n")
        f.close()
