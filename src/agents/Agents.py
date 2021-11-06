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
from src.database.Models import getMeasuresORM, getMeasureORM, getSensorORM, Measure, getObservations
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
    df = pd.DataFrame(data=measure_tups, columns=Measure._attr_names)
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

        return preped, attributes
    removed = list(set(obs) - set(columns))
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


def createDataframe(measures, columns):
    df = pd.DataFrame(data=measures, columns=columns)
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
               "completeness": 0.75}

    def __init__(self, sensor_id, config=None, confidence=50):
        self.cf = confidence
        if config:
            self.configs = config

        self.sensor = getSensorORM(sensor_id)
        self.sensor_list = []
        # self.pred_tile = target_tile
        # self.target_time = target_time

    def __iter__(self):
        for attr in self._attr_names:
            yield getattr(self, attr)

    @property
    def _attr_names(self):
        return [attr_name for attr_name in self.__dict__ if '_' not in attr_name]

    @abstractmethod
    def makePrediction(self, target_sensor, time, n=1, *values):
        pass

    def getConfigKeys(self):
        return ["sensor","agent","trust","configs"]

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
            sorted_measures.extend(self._fillInterval(start, start + datetime.timedelta(hours=hours - 1), measure_sid))

        if last_date < end:
            hours = self._countInterval(last_date, end)
            sorted_measures.extend(
                self._fillInterval(last_date + one_hour, datetime.timedelta(hours=hours), measure_sid))

        return sorted(data, key=lambda x: x.date)

    def _rateInterval(self, dataset, total):
        return (dataset - 1) / total

    def _impute_missing(self, measures):
        """

        :param measures: list of tuples containing Measure attributes
        :return: list of imputed values based of average of same values for each hour of dataset
        """
        imputed_data = list()
        if not measures:
            return []

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

        return sorted(measures, key=lambda m: m[1])

    def _prepareData(self, target_sensor, prediction_time, day_interval=0, hour_interval=1, targetObs: [] = None):
        """
        :param target_sensor: integer sensor id
        :param prediction_time: datetime object defining the time of prediction.
        :param day_interval: amount of historical data to use from the time of prediction by day.
        :param hour_interval: amount of historical data to use from the time of prediction by hour.
        :return: list of tuples representing measurement objects with no time gaps.
        """
        end_offset = datetime.timedelta(hours=1)
        time_interval = datetime.timedelta(days=day_interval, hours=hour_interval)
        new_end = (prediction_time - end_offset)
        new_start = new_end - time_interval
        with Session as sesh:
            orm_data = sesh.query(Measure).filter(Measure.date >= new_start).filter(Measure.date <= new_end). \
                filter(Measure.sid == target_sensor).all()

        orm_data = sorted(orm_data, key=lambda x: x.date)

        total_hours = self._countInterval(new_end, prediction_time)
        complete = len(orm_data) / total_hours
        if len(orm_data) == total_hours:
            # nothing to clean
            m_tuples, field_names = new_prepMeasures(orm_data, columns=targetObs)
            return m_tuples, field_names

        if complete < 0.75:
            # too much data missing for interval
            return None, None

        cleaned = self._cleanIntervals(orm_data, new_start, new_end)
        # df = measure_to_df(cleaned)
        measure_tuples, fields_order = new_prepMeasures(cleaned, columns=targetObs)
        imputed = self._impute_missing(measure_tuples)

        return imputed, fields_order


# predict value between 0 and maximum value of target sensor
class RandomAgent(Agent):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, time, *values, n=1):
        vals = self.validate_measures(values)
        data, cols = self._prepareData(target_sensor, time, targetObs=vals)
        max_vals = []
        for target_measure in values:
            max_vals.append((target_measure, max(data, key=lambda measure: measure[measure.index(target_measure)])))

        result = [(tup[0], rand.randint(0, tup[1])) for tup in max_vals]

        return result


# avg of nearby sensors to target sensor
class NearbyAverage(Agent):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, target_time, *values, n=1):
        target_predictions = self.validate_measures(values)
        sensors = findNearestSensors(self.sensor.sid, self.sensor_list)
        totals = {field: 0 for field in target_predictions}
        n = self.configs["n"]

        for s in sensors[:n]:
            data, cols = self._prepareData(s[0].sid, target_time, targetObs=target_predictions)
            if not data:
                return []
            latest_measure = data[-1]
            for field in target_predictions:
                totals[field] += latest_measure[cols.index(field)]

        return [(key, totals[key] / n) for key in totals.keys()]

    def getConfigKeys(self):
        return []


# average from min/max of nearby sensors to target sensor
class MinMaxAgent(Agent):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, time, *values, n=1):
        target_predictions = self.validate_measures(values)
        sensor_dists = findNearestSensors(target_sensor, self.sensor_list)
        sensor_vals = {val: [] for val in target_predictions}
        n = self.configs["n"]

        for sd in sensor_dists[:n]:
            data, cols = self._prepareData(sd[0].sid, time, targetObs=target_predictions)
            latest_measure = data[-1]
            for field in target_predictions:
                sensor_vals[field].append(latest_measure[cols.index(field)])

        results = []
        for k in sensor_vals.keys():
            max_m = max(sensor_vals[k])
            min_m = min(sensor_vals[k])
            results.append((k, (max_m + min_m) / len(sensor_vals[k])))

        return results


# value of nearest station
class NearestSensor(Agent):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, time, *values, n=1):
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        target_predictions = self.validate_measures(values)
        hour = datetime.timedelta(hours=1)
        measure_time = time - hour
        sensor_vals = {val: 0 for val in target_predictions}
        sensor_id = sensors[0][0].sid

        data, cols = self._prepareData(sensor_id, time, targetObs=target_predictions)
        latest_measure = data[-1]
        for field in target_predictions:
            sensor_vals[field] = latest_measure[cols.index(field)]

        results = []
        for k in sensor_vals.keys():
            results.append((k, sensor_vals[k]))

        return results


# Weighted Moving Average
class WmaAgent(Agent):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, time, *values, n=1 ):
        window = 10
        orm_data = getMeasuresORM(target_sensor)
        # most recent gets higher weight.
        weights = np.array(exp_weights(window))
        cols, data = prepareMeasures(orm_data, "pm1")
        df = createDataframe(cols, data)
        df.sort_values(by='date')
        pred = df['pm1'].rolling(window).apply(lambda x: np.sum(weights * x))
        # plt.plot(df['date'], df['pm1'], label="PM1 Values")
        # plt.plot(df['date'], pred, label="Pred")
        # plt.xlabel('Dates')
        # plt.ylabel('Values')
        # plt.legend()
        # plt.show()
        return pred


class CmaAgent(Agent):
    def __init__(self, sensor_id, config=None, confidence=50, cma=False):
        super().__init__(sensor_id, config, confidence=confidence)
        self._include_cma = cma

    def makePrediction(self, target_sensor, time, *values, n=1):
        window = 10
        orm_data = getMeasuresORM(target_sensor)
        col, data = prepareMeasures(orm_data, "pm1")
        results = createDataframe(col, data)
        results['Prediction'] = results.pm1.rolling(window, min_periods=1).mean()
        if self._include_cma:
            results['Cma'] = results.pm1.expanding(20).mean()
            return [('prediction', results['Prediction'][-1]), ('cma', results['Cma'][-1])]
            # plt.plot(results['date'], results['Prediction_cma'], label="Pred_cma")

        results['Error'] = abs(((results['pm1'] - results['Prediction']) / results['pm1']) * 100)
        # plt.plot(results['date'], results['pm1'], label="PM1 Values")
        # plt.plot(results['date'], results['Prediction'], label="Pred_ma")
        # plt.xlabel('Dates')
        # plt.ylabel('Values')
        # plt.legend()
        # plt.show()
        return [('prediction', results['Prediction'][-1])]


# To Do: AutoREgressiveIntegratedMovingAverage
class ARMIAX(Agent):
    def __init__(self, sensor_id, config=None, stationary=True):
        super().__init__(sensor_id, config=config)
        self.seasonal = not stationary

    def makePrediction(self, target_sensor, time, *values, n=1):

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
class MultiVariate(Agent):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, prediction_time, *values):
        days = self.configs["interval"]["days"]
        hours = self.configs["interval"]["hours"]
        if not days:
            days = 7
            hours = 0

        target_obs = self.validate_measures(values)
        data, columns = self._prepareData(target_sensor, prediction_time, day_interval=days, hour_interval=hours)
        predictions = { ob: 0.0 for ob in target_obs}
        # multi-step prediction
        # for hour in range(forward_hours):
        #     time_increment = datetime.timedelta(hours=hour)
        actual_value = getMeasureORM(target_sensor, prediction_time)

        for ob in target_obs:
            # fetch list of the other dependent variables: temp, pmN, pmN
            dependent_vars = getObservations(exclude=ob)
            independent_i = columns.index(ob)
            X = []
            Y = []
            for row in data:
                dvs = list()
                for dv in dependent_vars:
                    dvs.append(row[columns.index(dv)])
                X.append(dvs)
                Y.append(row[independent_i])

            reg_model = linear_model.LinearRegression()
            reg_model.fit(X, Y)
            # TODO: use MA to estimate dependent variables for future hours, if data is not available in database.
            test_data = [getattr(actual_value, attr) for attr in dependent_vars]
            predictions[ob] = reg_model.predict([test_data])

        return predictions


def findModelInterval(sid, time, interval_size):
    measure_list = getMeasuresORM(sid, end_interval=time)
    if len(measure_list) > interval_size:
        measure_list = sorted(measure_list, key=lambda x: x.date, reverse=False)
        model_list = measure_list[-interval_size:]
        return model_list[0], time
    else:
        return 0, 0


if __name__ == '__main__':
    # m = Measure(12345, 1, datetime.datetime(2021,9, 25, 18), 1.2, 25.2, 10.2, 12)
    pms = ['pm1']
    randy = NearbyAverage(11583)
    target_time = datetime.datetime(year=2019, month=1, day=7, hour=0)
    vals = randy.validate_measures(pms)
    # "start_interval": "2019-01-01 00:00",
    # "end_interval": "2019-01-02 00:00",
    data, cols = randy._prepareData(randy.sensor.sid, target_time, targetObs=vals)
    # prediction = randy.makePrediction(11594, target_time)
    print(cols)
    for d in data:
        print(d)

    print(data[-1])
    # print(prediction)

    # neary = NearbyAverage(11594)
    # pred = neary.makePrediction(11594, target_time, 'pm1', n=1)

    # print([x for x in c if not str(x).startswith("_")])
