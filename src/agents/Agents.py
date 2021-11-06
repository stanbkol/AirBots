from ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate


class Agent(object):
    def __init__(self, sensor_id, thresholds, config=None, confidence=100):
        self.sid = sensor_id
        self._threshold = thresholds
        self._configs = config
        self.confidence = confidence
        self.models = self._initializeModels()
        self.error = 0

    def _initializeModels(self):
        models = {"rand": RandomModel(self.sid),
                  'nearby': NearbyAverage(self.sid),
                  'minmax': MinMaxModel(self.sid),
                  'sma': CmaModel(self.sid),
                  'mvr': MultiVariate(self.sid)
                  }

        return models

    def _getModelNames(self):
        return ['rand', 'nearby', 'minmax', 'sma', 'mvr']

    def _weightModels(self, actual):
        errors = {}
        total_se = 0
        for model_name in self._getModelNames():
            model_prediction = self.models[model_name].getPrediction()
            squared_error = self._calc_squared_error(model_prediction, actual)
            total_se += squared_error
            errors[model_name] = squared_error

        return {model_name: errors[model_name]/total_se for model_name in errors.keys()}

    def _calc_error(self, prediction, actual):
        return (actual-prediction)/actual

    def _calc_squared_error(self, prediction, actual):
        return (actual-prediction)**2

    def _makePrediction(self):
        for model in self._getModelNames():
            self.models[model].makePrediction(self.sid, )


if __name__ == '__main__':
   pass
