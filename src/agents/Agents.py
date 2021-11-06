from src.agents.ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate


class Agent(object):
    def __init__(self, sensor_id, thresholds, sensor_list, config=None, confidence=100):
        self.sid = sensor_id
        self._threshold = thresholds
        self._configs = config
        self.confidence = confidence
        self.sensors = sensor_list
        self.models = self._initializeModels()
        self.error = 0

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
            squared_error = self._calc_squared_error(model_prediction['pm1'], actual)
            total_se += squared_error
            errors[model_name] = squared_error

        return {model_name: errors[model_name]/total_se for model_name in errors.keys()}

    def _calc_error(self, prediction, actual):
        return (actual-prediction)/actual

    def _calc_squared_error(self, prediction, actual):
        return (actual-prediction)**2

    def makePredictions(self, target_sid, target_time, values, meas=None):
        measure = meas
        predicts = {}
        for model in self._getModelNames():
            if model == 'mvr':
                predicts[model] = self.models[model].makePrediction(target_time, values, target_sid)
            else:
                predicts[model] = self.models[model].makePrediction(target_time, values)

        weights = self._weightModels(predicts, getattr(measure, 'pm1'))
        total_pm1 = 0
        for model in weights.keys():
            total_pm1 += weights[model] * predicts[model]['pm1']

        print(f'Sensor %s, ')
        return total_pm1


if __name__ == '__main__':
   pass
