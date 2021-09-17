import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from edu.pwr.database.utils import drange


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


def weighted_rolling_mean(data, window, cols, exp=True):
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


def arima_model(p, d, w):
    pass


def calc_weights(n):
    n += 1
    diff = 1 / n
    sample = [x for x in drange(0, 1, diff)][1::]
    # print(sample)
    total = sum(sample)
    # print(total)
    return [c / total for c in sample]


def exp_weights(n):
    if n < 2:
        return [n]
    r = (1 + n ** 0.5) / 2
    total = 1
    a = total * (1 - r) / (1 - r ** n)
    return [a * r ** i for i in range(n)]


def spatial_weights(sensor, n):
    neighbors = sensor.nearest_neighbors(n)
    s = 0
    d = 1
    sum_dist = sum(n[d] for n in neighbors)

    # normalized dist weights map(sid->dist_weight)
    sid_dist = {n[s].sid: (n[d]/sum_dist) for n in neighbors}

    return sid_dist


def getPM1(conn, sid, start_interval=None, end_interval=None):
    with conn.cursor() as cursor:
        if not start_interval and not end_interval:
            query = 'SELECT * FROM dbo.Measurements WHERE sensorID = %s;'
            data = [sid]
        else:
            query = 'SELECT sensorid, date, pm1 FROM dbo.Measurements WHERE sensorID = %s AND date BETWEEN %s AND %s;'
            data = [sid, start_interval, end_interval]
        cursor.execute(query, data)
        cols = []
        for elt in cursor.description:
            cols.append(elt[0])
        data_list = cursor.fetchall()
        return cols, data_list


def show_wma(conn, start, end):
    cols, measures = getPM1(conn, 11563, start, end)
    conn.close()
    res = weighted_rolling_mean(measures, 24, cols, exp=True)
    plt.plot(res['date'], res['pm1'], label="PM1 Values")
    plt.plot(res['date'], res['MA_PM1'], label="MA Values")
    plt.xlabel('Dates')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

def run():
    xp = exp_weights(2)
    print(xp)
    print(sum(xp))


if __name__ == '__main__':
    run()
