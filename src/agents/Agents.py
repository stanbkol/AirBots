from abc import ABC, abstractmethod
import random as rand

import pandas
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy.exc import NoResultFound

from src.database.utils import drange
import statsmodels.api as sm
from src.database.Models import getMeasuresORM, getMeasureORM, getSensorORM, Measure
from src.map.MapPoint import MapPoint, calcDistance


# # or maybe this design?
# class naiveAgent(Agent):
#     def __init__(self, prediction_method):
#         super().__init__()
#         self.prediction = prediction_method
#
#     def makePrediction(self, orm_data):
#         self.prediction(orm_data)


def findNearestSensors(sensorid, s_list):
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

    return distances


# to be updated
def prepareMeasures(dataset, col='*'):
    columns = []
    measure_data = []
    if col == "pm1":
        print("Fetching pm1 measures..")
        for entry in dataset:
            measure_data.append((entry.sid, entry.date, entry.pm1))
            columns = ['sensorid', 'date', 'pm1']
    if col == "pm10":
        print("Fetching pm10 measures..")
        for entry in dataset:
            measure_data.append((entry.sid, entry.date, entry.pm10))
            columns = ['sensorid', 'date', 'pm10']

    if col == "pm25":
        print("Fetching pm25 measures..")
        for entry in dataset:
            measure_data.append((entry.sid, entry.date, entry.pm25))
            columns = ['sensorid', 'date', 'pm25']

    if col == "temp":
        print("Fetching temperature measures..")
        for entry in dataset:
            measure_data.append((entry.sid, entry.date, entry.temp))
            columns = ['sensorid', 'date', 'temp']
    else:
        for entry in dataset:
            measure_data.append((entry.sid, entry.date, entry.temp, entry.pm1, entry.pm10, entry.pm25))
            columns = ['sensorid', 'date', 'temp', 'pm1', 'pm10', 'pm25']
    return columns, measure_data


def createDataframe(cols, measures):
    df = pd.DataFrame(data=measures, columns=cols)
    return df.sort_values(by="date", ascending=True)


def getColumn(dataset, col):
    column = []
    if col == "pm1":
        print("Fetching pm1 measures..")
        for entry in dataset:
            column.append(entry.pm1)
    if col == "pm10":
        print("Fetching pm10 measures..")
        for entry in dataset:
            column.append(entry.pm10)

    if col == "pm25":
        print("Fetching pm25 measures..")
        for entry in dataset:
            column.append(entry.pm25)

    if col == "temp":
        print("Fetching temperature measures..")
        for entry in dataset:
            column.append(entry.temp)
    return column


def exp_weights(n):
    if n < 2:
        return [n]
    r = (1 + n ** 0.5) / 2
    total = 1
    a = total * (1 - r) / (1 - r ** n)
    return [a * r ** i for i in range(n)]


def calc_weights(n):
    """

    :param n:
    :return:
    """
    n += 1
    diff = 1 / n
    sample = [x for x in drange(0, 1, diff)][1::]
    # print(sample)
    total = sum(sample)
    # print(total)
    return [c / total for c in sample]


class Agent(object):
    """

    """
    configs = {"error": 0.75,
               "confidence_error": 0.35,
               }

    def __init__(self, sensor_id, config=None, confidence=50):
        self.cf = confidence
        if config:
            self.configs = config

        self.sensor = getSensorORM(sensor_id)
        self.sensor_list = []
        # self.pred_tile = target_tile
        # self.target_time = target_time

    @abstractmethod
    def makePrediction(self, target_sensor, time, n=1, *values):
        pass

    # initial confidence calculation
    # self.cf += updateConfidence(self.cf, actual_value=, predicted_value=) <---in each make prediction method
    def _updateConfidence(self, predicted_value, actual_value):
        delta = (predicted_value - actual_value) / actual_value
        if delta < 0.1:
            return self.cf + 1
        else:
            return self.cf - 1

    def _countInterval(self, start, end):
        diff = end - start
        days, seconds = diff.days, diff.seconds
        total_intervals = days * 24 + seconds // 3600
        return total_intervals

    def _fillInterval(self, startMeasure, endMeasure):
        avg_pm1 = (startMeasure.pm1 + endMeasure.pm1) / 2
        avg_pm10 = (startMeasure.pm10 + endMeasure.pm10) / 2
        avg_pm25 = (startMeasure.pm25 + endMeasure.pm25) / 2
        avg_temp = (startMeasure.temp + endMeasure.temp) / 2
        time_range = pandas.date_range(startMeasure.date, endMeasure.date, freq='H')
        estimated_data = []
        for e in time_range:
            dk = int(e.strftime('%Y%m%d%H'))
            estimated_data.append(
                Measure(date_key=dk, sensor_id=startMeasure.sid, date=e, pm1=avg_pm1, pm10=avg_pm10, pm25=avg_pm25,
                        temperature=avg_temp))
        estimated_data.pop(0)
        estimated_data.pop()
        return estimated_data

    def _cleanInterval(self, data):
        for first, second in zip(data, data[1:]):
            if (self._countInterval(first.date, second.date)) != 1:
                data.extend(self._fillInterval(first, second))
        return sorted(data, key=lambda x: x.dk)

    def _rateInterval(self, dataset, total):
        return (dataset - 1) / total

    def _prepareData(self, target_sensor, time, values):
        return None


# predict value between 0 and maximum value of target sensor
class randomAgent(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        data = getMeasuresORM(target_sensor)
        max_m = max(data, key=lambda item: item.pm1)
        return rand.uniform(0, max_m.pm1)


# avg of nearby sensors to target sensor
class simpleAgentV1(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        total = 0
        n = self.configs["n"]
        for s in sensors[:n]:
            print(s)
            try:
                total += getMeasureORM(s[0].sid, time).pm1
            except NoResultFound:
                print("No data for sensor")
        return total / n


# average from min/max of nearby sensors to target sensor
class simpleAgentV2(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        sensor_vals = []
        n = self.configs["n"]
        for s in sensors[:n]:
            print(s)
            try:
                sensor_vals.append(getMeasureORM(s[0].sid, time))
            except NoResultFound:
                print("No data for sensor")
        try:
            max_m = max(sensor_vals, key=lambda item: item.pm1)
            min_m = min(sensor_vals, key=lambda item: item.pm1)
            return (max_m.pm1 + min_m.pm1) / n
        except ValueError:
            return 0


# value of nearest station
class simpleAgentV3(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        values = []
        for s in sensors:
            try:
                values.append(getMeasureORM(s, time).pm1)
            except:
                values.append(None)
        print(values)
        try:
            return next(item for item in values if item is not None)
        except StopIteration:
            return 0


# Weighted Moving Average
class MovingAverageV1(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        window = 10
        orm_data = getMeasuresORM(target_sensor)
        weights = np.array(exp_weights(window))
        cols, data = prepareMeasures(orm_data, "pm1")
        df = createDataframe(cols, data)
        df.sort_values(by='date')
        pred = df['pm1'].rolling(window).apply(lambda x: np.sum(weights * x))
        plt.plot(df['date'], df['pm1'], label="PM1 Values")
        plt.plot(df['date'], pred, label="Pred")
        plt.xlabel('Dates')
        plt.ylabel('Values')
        plt.legend()
        plt.show()


class MovingAverageV2(Agent):

    def makePrediction(self, target_sensor, time, n=1, *values):
        window = 10
        orm_data = getMeasuresORM(target_sensor)
        col, data = prepareMeasures(orm_data, "pm1")
        results = createDataframe(col, data)
        self.include_cma = True
        results['Prediction'] = results.pm1.rolling(window, min_periods=1).mean()
        if self.include_cma:
            results['Prediction_cma'] = results.pm1.expanding(20).mean()
            plt.plot(results['date'], results['Prediction_cma'], label="Pred_cma")

        results['Error'] = abs(((results['pm1'] - results['Prediction']) / results['pm1']) * 100)
        plt.plot(results['date'], results['pm1'], label="PM1 Values")
        plt.plot(results['date'], results['Prediction'], label="Pred_ma")
        plt.xlabel('Dates')
        plt.ylabel('Values')
        plt.legend()
        plt.show()


# To Do: AutoREgressiveIntegratedMovingAverage
class ARMIAX(Agent):
    # def __init__(self, stationary=True):
    #     super().__init__()
    #     self.seasonal = not stationary

    def makePrediction(self, target_sensor, time, n=1, *values):
        orm_data = self._prepareData(target_sensor, time, values)
        cols, measures = prepareMeasures(orm_data, '*')
        df = createDataframe(cols, measures)
        # ['sensorid', 'date', 'temp', 'pm1', 'pm10', 'pm25']

        # date, pm1
        series = df.drop(['sensorid', 'temp', 'pm10', 'pm25'], axis=1).dropna()
        # other related variables: temp and the other pms
        exog_train = df.drop(['sensorid', 'date', 'pm1'], axis=1).dropna()

        # TODO: These need to be found based on the data provided. see https://people.duke.edu/~rnau/411arim.htm
        ar_p = 5
        i_d = 1
        ma_q = 0

        if not self.seasonal:
            model = sm.tsa.statespace.SARIMAX(series, order=(ar_p, i_d, ma_q),
                                              seasonal_order=(0, 0, 0, 0),
                                              exog=exog_train)
        else:
            # TODO: seasonal_order cannot be (0,0,0,0) change it to something else.
            model = sm.tsa.statespace.SARIMAX(series, order=(ar_p, i_d, ma_q),
                                              seasonal_order=(0, 0, 0, 0),
                                              exog=exog_train)

        model_fit = model.fit()

        # returns the n+1 hour prediction for pm1
        return model_fit.forecast()[0]


# update to involve two sensorids; use the predicting sensors data to define the model, and plug in values from the
# target sensor to make prediction
class MultiDimensionV1(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        reg_model = linear_model.LinearRegression()
        interval = findModelInterval(self.sid, time, 100)
        model_data = getMeasuresORM(self.sid, interval[0], interval[1])
        x = list(zip(getColumn(model_data, 'temp'), getColumn(model_data, 'pm10'), getColumn(model_data, 'pm25')))
        y = getColumn(model_data, 'pm1')
        reg_model.fit(x, y)
        actual_value = getMeasureORM(target_sensor, time)
        prediction = reg_model.predict([[actual_value.temp, actual_value.pm10, actual_value.pm25]])
        return prediction


def findModelInterval(sid, time, interval_size):
    measure_list = getMeasuresORM(sid, end_interval=time)
    if len(measure_list) > interval_size:
        measure_list = sorted(measure_list, key=lambda x: x.date, reverse=False)
        model_list = measure_list[-interval_size:]
        return model_list[0], time
    else:
        return 0, 0
