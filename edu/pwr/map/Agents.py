from abc import ABC, abstractmethod, abstractproperty
from sklearn import linear_model
import numpy as np
import pandas as pd
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
