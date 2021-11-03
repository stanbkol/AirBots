import datetime
from abc import ABC, abstractmethod
import random as rand

import pandas
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy.exc import NoResultFound

from src.database.DbManager import Session
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

    return distances


def measure_to_df(measures_list):
    measure_tups = [tuple(m) for m in measures_list]
    df = pd.DataFrame(data=measure_tups, columns=Measure.attr_names)
    return df.sort_values(by="date", ascending=True)


def getAttributes(obj, attrs):
    for attr in attrs:
        yield getattr(obj, attr)


def new_prepMeasures(measure_list, columns=None):
    preped = []
    obs = ['temp', 'pm1', 'pm10', 'pm25']
    attributes = [attr_name for attr_name in Measure.__dict__ if not str(attr_name).startswith("_")]
    attributes.remove('dk')
    attributes.remove('Sensor')
    if not columns:
        for m in measure_list:
            preped.append(tuple(getAttributes(m, attributes)))

        return preped
    removed = list(set(obs) - set(columns))
    print(attributes)
    print(removed)
    wanted = [item for item in attributes if item not in removed]
    for measure in measure_list:
        m_tup = tuple(getAttributes(measure, wanted))
        preped.append(m_tup)

    return preped, wanted


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
               "completeness":0.75}

    def __init__(self, sensor_id, config=None, confidence=50):
        self.cf = confidence
        if config:
            self.configs = config

        self.sensor = getSensorORM(sensor_id)
        self.sensor_list = []
        # self.pred_tile = target_tile
        # self.target_time = target_time

    def __iter__(self):
        for attr in self.attr_names:
            yield getattr(self, attr)

    @property
    def attr_names(self):
        return [attr_name for attr_name in self.__dict__ if '_' not in attr_name]

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

    def validate_measures(self, observations):
        obs = ['pm1', 'pm10', 'pm25']
        vals = [target for target in set(observations) if target in obs]
        return vals

    def _countInterval(self, start, end):
        diff = end - start
        days, seconds = diff.days, diff.seconds
        total_intervals = days * 24 + seconds // 3600
        return total_intervals

    def _fillInterval(self, start_date, end_date, sid):
        time_range = pandas.date_range(start_date, end_date, freq='H')
        estimated_data = []
        for hour in time_range:
            dk = int(hour.strftime('%Y%m%d%H'))
            estimated_data.append(
                Measure(date_key=dk, sensor_id=sid, date=hour))
        return estimated_data

    def _cleanIntervals(self, data, start, end):
        """

        :param data: list of Measurement objects.
        :return: list of Measurement objects with filled in missing hours, sorted by date field
        """
        measure_sid = data[0].sid
        for first, second in zip(data, data[1:]):
            if (self._countInterval(first.date, second.date)) != 1:
                filled_interval = self._fillInterval(first.date, second.date, measure_sid)
                filled_interval.pop(0)
                filled_interval.pop()
                data.extend(filled_interval)

        sorted_measures = sorted(data, key=lambda x: x.date)

        first_date = sorted_measures[0].date
        last_date = sorted_measures[-1].date
        one_hour = datetime.timedelta(hours=1)
        if first_date > start:
            hours = self._countInterval(start, first_date)
            sorted_measures.extend(self._fillInterval(start, start+datetime.timedelta(hours=hours-1), measure_sid))

        if last_date < end:
            hours = self._countInterval(last_date, end)
            sorted_measures.extend(self._fillInterval(last_date + one_hour, datetime.timedelta(hours=hours), measure_sid))

        return sorted(data, key=lambda x: x.date)

    def _rateInterval(self, dataset, total):
        return (dataset - 1) / total

    def _impute_missing(self, measures):
        """

        :param measures: list of tuples containing Measure attributes
        :return: list of imputed values based of average of same values for each hour of dataset
        """
        imputed_data = list()
        dk = 0
        time = 2
        for m in measures:
            empties = [i for i, v in enumerate(m) if v is None]
            if empties:
                # every None index
                for i in empties:
                    all_rows = list()
                    all_rows.append(
                        [row[i] for row in measures if (row[dk] != m[dk] and m[time].hour == row[time].hour)])
                    median = np.median([x for x in all_rows if x is not None])
                    m[i] = median

        return measures

    def _prepareData(self, target_sensor, time, day_interval=7, targetObs: [] = None):
        """

        :param target_sensor: integer sensor id
        :param time: datetime object defining the time of prediction.
        :param day_interval: amount of historical data to use from the time of prediction.
        :return: list of tuples representing measurement objects with no time gaps.
        """
        hour = datetime.timedelta(hours=1)
        days = datetime.timedelta(days=day_interval)
        new_end = (time - hour)
        new_start = new_end - days
        with Session as sesh:
            orm_data = sesh.query(Measure).filter(Measure.date >= new_start).filter(Measure.date <= new_end). \
                filter(Measure.sid == target_sensor).all()

        orm_data = sorted(orm_data, key=lambda x: x.date)

        total_hours = self._countInterval(new_end, time)
        complete = len(orm_data)/total_hours
        if len(orm_data) == total_hours:
            # nothing to clean
            m_tuples, field_names = new_prepMeasures(orm_data, columns=targetObs)
            return m_tuples, field_names

        if complete < self.configs["completeness"]:
            # too much data missing for interval
            return None, None

        cleaned = self._cleanIntervals(orm_data, new_start, new_end)
        # df = measure_to_df(cleaned)
        measure_tuples, fields_order = new_prepMeasures(cleaned)
        imputed = self._impute_missing(measure_tuples)

        return imputed, fields_order


# predict value between 0 and maximum value of target sensor
class RandomAgent(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        vals = self.validate_measures(values)
        data, cols = self._prepareData(target_sensor, time, targetObs=vals)
        max_vals = []
        for target_measure in values:
            max_vals.append( (target_measure, max(data, key=lambda measure: measure[measure.index(target_measure)])))

        result = [(tup[0], rand.randint(0, tup[1])) for tup in max_vals]

        return result


# avg of nearby sensors to target sensor
class NearbyAverage(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        fields = self.validate_measures(values)
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        hour = datetime.timedelta(hours=1)
        measure_time = time - hour
        totals = {field:0 for field in fields}

        n = self.configs["n"]
        for s in sensors[:n]:
            print(s)
            try:
                measure = getMeasureORM(s[0].sid, measure_time)
                for obs in fields:
                    totals[obs] += getattr(measure, obs)
            except NoResultFound:
                print("No data for sensor")
        return [(key, totals[key]/n) for key in totals.keys()]


# average from min/max of nearby sensors to target sensor
class MinMaxAgent(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        fields = self.validate_measures(values)
        sensor_dists = findNearestSensors(target_sensor, self.sensor_list)
        hour = datetime.timedelta(hours=1)
        measure_time = time - hour
        sensor_vals = {val: [] for val in fields}
        n = self.configs["n"]

        for sd in sensor_dists[:n]:
            try:
                # measures, cols = self._prepareData()
                measure = getMeasureORM(sd[0].sid, measure_time)
                for field in fields:
                    sensor_vals[field].append(getattr(measure, field))
            except NoResultFound:
                print("No data for sensor")

        results = []
        for k in sensor_vals.keys():
            max_m = max(sensor_vals[k])
            min_m = min(sensor_vals[k])
            results.append((k, (max_m + min_m) / len(sensor_vals[k])))

        return results


# value of nearest station
class NearestSensor(Agent):
    def makePrediction(self, target_sensor, time, n=1, *values):
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        fields = self.validate_measures(values)
        hour = datetime.timedelta(hours=1)
        measure_time = time - hour
        sensor_vals = {val: [] for val in fields}
        for sd in sensors:
            try:
                measure = getMeasureORM(sd[0].sid, measure_time)
                for field in fields:
                    sensor_vals[field].append(getattr(measure, field))
            except:
                for field in fields:
                    sensor_vals[field].append(None)
        print(values)
        try:
            return next(item for item in values if item is not None)
        except StopIteration:
            return 0


# Weighted Moving Average
class WmaAgent(Agent):
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


class CmaAgent(Agent):
    def __init__(self, sensor_id, config=None, confidence=50):
        super().__init__(sensor_id, config, confidence)
        self._include_cma = False

    def makePrediction(self, target_sensor, time, n=1, *values):
        window = 10
        orm_data = getMeasuresORM(target_sensor)
        col, data = prepareMeasures(orm_data, "pm1")
        results = createDataframe(col, data)
        results['Prediction'] = results.pm1.rolling(window, min_periods=1).mean()
        if self._include_cma:
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
    def __init__(self, sensor_id, config=None, stationary=True):
        super().__init__(sensor_id, config=config)
        self.seasonal = not stationary

    def makePrediction(self, target_sensor, time, n=1, *values):
        orm_data, cols = self._prepareData(target_sensor, time)
        cols, measures = prepareMeasures(orm_data)
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
    def __init__(self, sensor_id, config=None, stationary=True):
        super().__init__(sensor_id, config=config)
        self.seasonal = not stationary

    def makePrediction(self, target_sensor, time, n=1, *values):
        reg_model = linear_model.LinearRegression()
        interval = findModelInterval(self.sensor.sid, time, 100)
        model_data = getMeasuresORM(self.sensor.sid, interval[0], interval[1])
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


if __name__ == '__main__':
    m = Measure(12345, 1, datetime.datetime(2021,9, 25, 18), 1.2, 25.2, 10.2, 12)
    # print(Measure.__dict__.keys())
    # print(dir(Measure))
    # print(Measure.attr_names())
    # print([a for a in vars(Measure) if not(a.startswith('_') and a.endswith('_'))])
    # print(vars(m))
    # print(tuple(m))

    a = [1,2,3,4]
    b = [3,2,1]
    c = ['_pm1', 'pm10']

    print([x for x in c if not str(x).startswith("_")])
    # print(list(set(a)-set(list(c))))
    # print(all(x in a for x in b))
