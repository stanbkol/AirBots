from abc import ABC, abstractmethod
from random import random

from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.database.utils import drange
import statsmodels.api as sm
from src.database.Models import getMeasuresORM, findNearestSensors, getMeasureORM

# # or maybe this design?
# class naiveAgent(Agent):
#     def __init__(self, prediction_method):
#         super().__init__()
#         self.prediction = prediction_method
#
#     def makePrediction(self, orm_data):
#         self.prediction(orm_data)


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


# initial confidence calculation
# self.cf += updateConfidence(self.cf, actual_value=, predicted_value=) <---in each make prediction method
def updateConfidence(cf, predicted_value, actual_value):
    delta = (predicted_value - actual_value) / actual_value
    if delta < 0.1:
        return cf + 1
    else:
        return cf - 1


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


class Agent(ABC):
    """"""""
    def __init__(self):
        self.cf = 50

    @abstractmethod
    def makePrediction(self, orm_data):
        pass


# predict value between 0 and maximum value of target sensor
class randomAgent(Agent):
    def makePrediction(self, sid):
        data = getMeasuresORM(sid)
        return random.uniform(0, max(data, key=lambda item: item.pm1))


# avg of nearby sensors to target sensor
class simpleAgentV1(Agent):

    def makePrediction(self, sid, time, n):
        sensors = findNearestSensors(sid)
        total = 0
        for s in sensors[:n]:
            total += getMeasureORM(s[0].sid, time).pm1
        return total / n


# average from min/max of nearby sensors to target sensor
class simpleAgentV2(Agent):
    def makePrediction(self, sid, time, n):
        sensors = findNearestSensors(sid)
        sensor_vals = []
        for s in sensors[:n]:
            sensor_vals.append(getMeasureORM(s[0].sid, time))
        return (max(sensor_vals, key=lambda item: item.pm1) + min(sensor_vals, key=lambda item: item.pm1)) / n


# value of nearest station
class simpleAgentV3(Agent):
    def makePrediction(self, sid, time):
        sensors = findNearestSensors(sid)
        return getMeasureORM(sensors[0]).pm1


# Weighted Moving Average
class MovingAverageV1(Agent):

    def makePrediction(self, orm_data):
        window = 10
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
    def __init__(self, cma=False):
        super().__init__()
        self.include_cma = cma

    def makePrediction(self, orm_data):
        window = 10
        col, data = prepareMeasures(orm_data, "pm1")
        results = createDataframe(col, data)
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
    def __init__(self, stationary=True):
        super().__init__()
        self.seasonal = not stationary

    def makePrediction(self, orm_data):
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
    def makePrediction(self, pred_sens, target_sens, time):
        reg_model = linear_model.LinearRegression()
        interval = findModelInterval(pred_sens, time, 100)
        model_data = getMeasuresORM(pred_sens, interval[0], interval[1])
        x = list(zip(getColumn(model_data, 'temp'), getColumn(model_data, 'pm10'), getColumn(model_data, 'pm25')))
        y = getColumn(model_data, 'pm1')
        reg_model.fit(x, y)
        actual_value = getMeasureORM(target_sens, time)
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
