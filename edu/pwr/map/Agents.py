from abc import ABC, abstractmethod, abstractproperty
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from edu.pwr.database.utils import drange


class Agent(ABC):

    @abstractmethod
    def makePrediction(self, data):
        pass

    @abstractproperty
    def confidence_factor(self):
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
def prepareMeasures(dataset, col):
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
    return columns, measure_data


def createDataframe(cols, measures):
    df = pd.DataFrame(data=measures, columns=cols)
    return df.sort_values(by="date", ascending=True)


# Exponential Moving Average
class MovingAverageV1(Agent):

    def makePrediction(self, data):
        pass

    def confidence_factor(self):
        pass

    def weighted_rolling_mean(self, data, window, cols, exp=True):
        if exp:
            weights = np.array(exp_weights(window))
        else:
            weights = np.array(calc_weights(window))

        print("length of weights: " + str(len(weights)))
        print(weights)
        print(sum(weights))
        df = pd.DataFrame(data=data, columns=cols)
        df.sort_values(by='date')
        df['MA_PM1'] = df['pm1'].rolling(window).apply(lambda x: np.sum(weights * x))
        return df


# Simple Moving Average
class MovingAverageV2(Agent):
    def makePrediction(self, orm_data):
        col, data = prepareMeasures(orm_data, "pm1")
        results = createDataframe(col, data)
        results['Prediction'] = results.pm1.rolling(10, min_periods=1).mean()
        results['Confidence'] = 100 - abs(((results['pm1'] - results['Prediction']) / results['pm1']) * 100)
        plt.plot(results['date'], results['pm1'], label="PM1 Values")
        plt.plot(results['date'], results['Prediction'], label="Pred")
        plt.xlabel('Dates')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def confidence_factor(self):
        pass


# Cumulative Moving Average
class MovingAverageV3(Agent):
    def makePrediction(self, orm_data):
        col, data = prepareMeasures(orm_data, "pm1")
        results = createDataframe(col, data)
        results['Prediction'] = results.pm1.rolling(10, min_periods=1).mean()
        results['Prediction_cma'] = results.pm1.expanding(20).mean()
        results['Error'] = abs(((results['pm1'] - results['Prediction']) / results['pm1']) * 100)
        plt.plot(results['date'], results['pm1'], label="PM1 Values")
        plt.plot(results['date'], results['Prediction'], label="Pred_ma")
        plt.plot(results['date'], results['Prediction_cma'], label="Pred_cma")
        # plt.plot(df2['date'], df2['pm1'], label="Prediction")
        plt.xlabel('Dates')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def confidence_factor(self):
        pass


class MultiDimensionV1(Agent):

    def makePrediction(self, data):
        pass

    def confidence_factor(self):
        pass

    def hypothesis(self, theta, X):
        return theta * X

    def computeCost(self, X, y, theta):
        y1 = self.hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        return sum(np.sqrt((y1 - y) ** 2)) / (2 * 47)

    def gradientDescent(self, X, y, theta, alpha, i):
        J = []  # cost function in each iterations
        k = 0
        while k < i:
            y1 = self.hypothesis(theta, X)
            y1 = np.sum(y1, axis=1)
            for c in range(0, len(X.columns)):
                theta[c] = theta[c] - alpha * (sum((y1 - y) * X.iloc[:, c]) / len(X))
            j = self.computeCost(X, y, theta)
            J.append(j)
            k += 1
        return J, j, theta

    '''-------------------------------------------------------'''

    # TODO: figure out how to create DataFrame using ORM query.
    def mv_reg_pred(self, df_data, temp, pm10, pm25):
        X = df_data[['temperature', 'pm10', 'pm25']]
        y = df_data['pm1']

        regr = linear_model.LinearRegression()
        regr.fit(X, y)

        prediction = regr.predict([[temp, pm10, pm25]])

        print(prediction)
