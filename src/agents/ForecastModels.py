import datetime
# from datetime import datetime, timedelta
from abc import abstractmethod
import random as rand
import pandas
import pandas as pd
from sklearn import linear_model
import numpy as np
from src.database.DbManager import Session
from src.database.utils import drange
import statsmodels.api as sm
from src.database.Models import getMeasureORM, getSensorORM, Measure, getObservations
from src.map.MapPoint import MapPoint, calcDistance


def createDataframe(measures, columns):
    df = pd.DataFrame(data=measures, columns=columns)
    return df.sort_values(by="date", ascending=True)


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
    total = sum(sample)
    return [c / total for c in sample]


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


def measure_to_df(measures_list, columns):
    measure_tups = [tuple(m) for m in measures_list]
    df = pd.DataFrame(data=measure_tups, columns=columns)
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


class Model(object):
    """

    """
    configs = {"error": 0.75,
               "confidence_error": 0.35,
               "completeness": 0.75}

    def __init__(self, sensor_id, sensor_list=None, config=None):
        if config:
            self.configs = config

        self.sensor = getSensorORM(sensor_id)
        self.sensor_list = sensor_list
        self._mse = 0
        # self.pred_tile = target_tile
        # self.target_time = target_time

    def __iter__(self):
        for attr in self._attr_names:
            yield getattr(self, attr)

    @property
    def _attr_names(self):
        return [attr_name for attr_name in self.__dict__ if '_' not in attr_name]

    @abstractmethod
    def makePrediction(self, time, values, n=1):
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
class RandomModel(Model):
    def __init__(self, sensor_id, sensor_list, config=None):
        super().__init__(sensor_id, sensor_list=sensor_list, config=config)

    def makePrediction(self, time, values, n=1):
        vals = self.validate_measures(values)
        data, cols = self._prepareData(self.sensor.sid, time, targetObs=vals, hour_interval=48)
        max_vals = {ob: 0 for ob in vals}
        for target_measure in vals:
            max_vals[target_measure] = max(data, key=lambda measure: measure[cols.index(target_measure)])

        return {ob: rand.uniform(0, max_vals[ob][2]) for ob in max_vals.keys()}


# avg of nearby sensors to target sensor
class NearbyAverage(Model):
    def __init__(self, sensor_id, sensor_list, config=None):
        super().__init__(sensor_id, sensor_list=sensor_list, config=config)

    def makePrediction(self, target_time, values, n=1):
        target_predictions = self.validate_measures(values)
        sensors = findNearestSensors(self.sensor.sid, self.sensor_list)
        totals = {field: 0 for field in target_predictions}
        n = 3
        for s in sensors[:n]:
            data, cols = self._prepareData(s[0].sid, target_time, targetObs=target_predictions)
            if not data:
                return []
            latest_measure = data[-1]
            for field in target_predictions:
                totals[field] += latest_measure[cols.index(field)]

        return {key: (totals[key] / n) for key in totals.keys()}

    def getConfigKeys(self):
        return []


# average from min/max of nearby sensors to target sensor
class MinMaxModel(Model):
    def __init__(self, sensor_id, sensors, config=None):
        super().__init__(sensor_id, sensor_list=sensors, config=config)

    def makePrediction(self, time, values, n=1):
        target_predictions = self.validate_measures(values)
        sensor_dists = findNearestSensors(self.sensor.sid, self.sensor_list)
        sensor_vals = {val: [] for val in target_predictions}
        n = 3
        for sd in sensor_dists[:n]:
            data, cols = self._prepareData(sd[0].sid, time, targetObs=target_predictions)
            latest_measure = data[-1]
            for field in target_predictions:
                sensor_vals[field].append(latest_measure[cols.index(field)])

        results = {}
        for k in sensor_vals.keys():
            max_m = max(sensor_vals[k])
            min_m = min(sensor_vals[k])
            results[k] = (max_m + min_m) / len(sensor_vals[k])

        return results


# value of nearest station
class NearestSensor(Model):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, time, *values, n=1):
        sensors = findNearestSensors(target_sensor, self.sensor_list)
        target_predictions = self.validate_measures(values)
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
class WmaModel(Model):
    def __init__(self, sensor_id, config=None):
        super().__init__(sensor_id, config=config)

    def makePrediction(self, target_sensor, time, *values, n=1 ):
        window = 10
        target_predictions = self.validate_measures(values)
        # most recent gets higher weight.
        weights = np.array(exp_weights(window))
        data, cols = self._prepareData(target_sensor, time, targetObs=target_predictions)
        df = createDataframe(cols, data)
        df.sort_values(by='date')
        pred = df['pm1'].rolling(window).apply(lambda x: np.sum(weights * x))
        return pred[-1]


class CmaModel(Model):
    def __init__(self, sensor_id, sensors, config=None, cma=False):
        super().__init__(sensor_id, sensor_list=sensors, config=config)
        self._include_cma = cma

    def makePrediction(self, time, values, window=10):
        target_predictions = self.validate_measures(values)
        data, cols = self._prepareData(self.sensor.sid, time, targetObs=target_predictions, hour_interval=48)
        results = createDataframe(data, cols)
        predictions = {ob: 0.0 for ob in target_predictions}

        for ob in target_predictions:
            results['ma_'+str(ob)] = results[ob].rolling(window, min_periods=1).mean()
            predictions[ob] = results['ma_'+str(ob)].iloc[-1]

        if self._include_cma:
            pass
            # for ob in target_predictions:
            #     results['Cma'] = results.pm1.expanding(20).mean()
            #     return [('prediction', results['Prediction'][-1]), ('cma', results['Cma'][-1])]
            # plt.plot(results['date'], results['Prediction_cma'], label="Pred_cma")

        # results['Error_'] = abs(((results['pm1'] - results['Prediction']) / results['pm1']) * 100)

        return predictions


# To Do: AutoREgressiveIntegratedMovingAverage
class ARMIAX(Model):
    def __init__(self, sensor_id, config=None, stationary=True):
        super().__init__(sensor_id, config=config)
        self.seasonal = not stationary

    def makePrediction(self, target_sensor, time, *values, n=1):

        measures, cols = self._prepareData(target_sensor, time)
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
class MultiVariate(Model):
    def __init__(self, sensor_id, sensors, config=None):
        super().__init__(sensor_id, sensor_list=sensors, config=config)

    def makePrediction(self, prediction_time, values, target_sid):
        # days = self.configs["interval"]["days"]
        # hours = self.configs["interval"]["hours"]
        # if not days:
        days = 7
        hours = 0

        target_obs = self.validate_measures(values)
        data, columns = self._prepareData(self.sensor.sid, prediction_time, day_interval=days, hour_interval=hours)
        predictions = { ob: 0.0 for ob in target_obs}
        # multi-step prediction
        # for hour in range(forward_hours):
        #     time_increment = datetime.timedelta(hours=hour)
        actual_value = getMeasureORM(target_sid, prediction_time)

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