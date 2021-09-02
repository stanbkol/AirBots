import numpy as np
import pandas as pd


def weightedmovingaverage(Data, period):
    weighted = []
    for i in range(len(Data)):
        try:
            total = np.arange(1, period + 1, 1)  # weight matrix
            matrix = Data[i - period + 1: i + 1, 3:4]
            matrix = np.ndarray.flatten(matrix)
            matrix = total * matrix  # multiplication
            wma = (matrix.sum()) / (total.sum())  # WMA
            weighted = np.append(weighted, wma)  # add to array
        except ValueError:
            pass
    return weighted


def arima_model(p, d, w):
    pass


def run():
    pass


if __name__ == '__main__':
    run()
