from src.agents.ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate
from src.database.DbManager import Session
from src.database.Models import fetchTile_from_sid


def _calc_error(prediction, actual):
    return (actual-prediction)/actual


def _calc_squared_error(prediction, actual):
    return (actual-prediction)**2


class Agent(object):
    def __init__(self, sensor_id, thresholds, sensor_list, config=None, confidence=100):
        self.sid = sensor_id
        self._threshold = thresholds
        self._configs = config
        self.cf = confidence
        self.sensors = sensor_list
        self.models = self._initializeModels()
        self.error = 0
        self.tile = fetchTile_from_sid(self.sid)

    def _initializeModels(self):
        models = {"rand": RandomModel(self.sid, self.sensors),
                  'nearby': NearbyAverage(self.sid, self.sensors),
                  'minmax': MinMaxModel(self.sid, self.sensors),
                  'sma': CmaModel(self.sid, self.sensors),
                  'mvr': MultiVariate(self.sid, self.sensors)
                  }

        return models

    def _getModelNames(self):
        return ['rand', 'nearby', 'minmax', 'sma', 'mvr']

    def _weightModels(self, predicts, actual):
        errors = {}
        total_se = 0
        for model_name in self._getModelNames():
            model_prediction = predicts[model_name]
            squared_error = _calc_squared_error(model_prediction['pm1'], actual)
            total_se += squared_error
            errors[model_name] = squared_error

        return {model_name: errors[model_name]/total_se for model_name in errors.keys()}

    def _updateConfidence(self, predicted_value, actual_value):
        delta = (predicted_value - actual_value) / actual_value
        if delta < 0.1:
            return self.cf + 1
        else:
            return self.cf - 1

    def tiles_change_factor(self, target_tile):
        path = self.tile.pathTo(target_tile)
        total = 0
        for p_i in range(1, len(path)):
            total += path[p_i].getCF() - path[p_i-1].getCF()

        return total

    def makePredictions(self, target_sid, target_time, values, meas=None):
        measure = meas
        predicts = {}
        for model in self._getModelNames():
            if model == 'mvr':
                predicts[model] = self.models[model].makePrediction(target_time, values, target_sid=target_sid)
            else:
                predicts[model] = self.models[model].makePrediction(target_time, values)

        weights = self._weightModels(predicts, getattr(measure, 'pm1'))
        total_pm1 = 0
        for model in weights.keys():
            total_pm1 += weights[model] * predicts[model]['pm1']

        print(f'Sensor %s, ' % self.sid)
        return total_pm1


if __name__ == '__main__':
   pass
