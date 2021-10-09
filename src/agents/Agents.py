from abc import ABC, abstractmethod
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.database.utils import drange
import statsmodels.api as sm


# initial confidence calculation
# self.cf += updateConfidence(self.cf, actual_value=, predicted_value=) <---in each make prediction method
def updateConfidence(cf, predicted_value, actual_value):
    delta = (predicted_value - actual_value) / actual_value
    if delta < 0.1:
        return cf + 1
    else:
        return cf - 1


class Agent(ABC):

    def __init__(self):
        self.cf = 50

    @abstractmethod
    def makePrediction(self, orm_data):
        pass


def exp_weights(n):
    if n < 2:
        return [n]
    r = (1 + n ** 0.5) / 2
    total = 1
    a = total * (1 - r) / (1 - r ** n)
    return [a * r ** i for i in range(n)]


def calc_weights(n):
    n += 1
    diff = 1 / n
    sample = [x for x in drange(0, 1, diff)][1::]
    # print(sample)
    total = sum(sample)
    # print(total)
    return [c / total for c in sample]


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


# or maybe this design?
class naiveAgent(Agent):
    def __init__(self, prediction_method):
        super().__init__()
        self.prediction = prediction_method

    def makePrediction(self, orm_data):
        self.prediction(orm_data)


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


class MultiDimensionV1(Agent):
    def makePrediction(self, orm_data):
        regr = linear_model.LinearRegression()
        x = list(zip(getColumn(orm_data, 'temp'), getColumn(orm_data, 'pm10'), getColumn(orm_data, 'pm25')))
        y = getColumn(orm_data, 'pm1')
        regr.fit(x, y)
        prediction = regr.predict([[-2.77, 207.3, 135.18]])
        print(prediction)