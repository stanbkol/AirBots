import numpy as np
import pandas as pd
from sklearn import linear_model

'''STOLEN FROM THE WEB'''
def hypothesis(theta, X):
    return theta*X


def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1=np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*47)


def gradientDescent(X, y, theta, alpha, i):
    J = []  # cost function in each iterations
    k = 0
    while k < i:
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(0, len(X.columns)):
            theta[c] = theta[c] - alpha*(sum((y1-y)*X.iloc[:,c])/len(X))
        j = computeCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta
'''-------------------------------------------------------'''

# TODO: figure out how to create DataFrame using ORM query.
def mv_reg_pred(df_data, temp, pm10, pm25):
    X = df_data[['temperature', 'pm10', 'pm25']]
    y = df_data['pm1']

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    prediction = regr.predict([[temp, pm10, pm25]])

    print(prediction)

