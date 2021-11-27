import math

from src.agents.ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate
from src.database.DbManager import Session
from src.database.Models import fetchTile_from_sid, Tile, getClassTiles


def _calc_error(prediction, actual):
    return (actual-prediction)/actual


def _calc_squared_error(prediction, actual):
    return (actual-prediction)**2


class Agent(object):
    def __init__(self, sensor_id, thresholds, cluster_list, config=None, confidence=100):
        self.sid = sensor_id
        self._threshold = thresholds
        self._configs = config
        self.cf = confidence
        self.cluster = cluster_list
        self.models = self._initializeModels()
        self.error = 0
        self.n_error = 0
        self.p_error = 0
        self.target_tile = None
        self.tile = fetchTile_from_sid(self.sid)
        self.bias = 0.70
        self.prediction = 0

    def _initializeModels(self):
        models = {"rand": RandomModel(self.sid, self.cluster),
                  'nearby': NearbyAverage(self.sid, self.cluster),
                  'minmax': MinMaxModel(self.sid, self.cluster),
                  'sma': CmaModel(self.sid, self.cluster),
                  'mvr': MultiVariate(self.sid, self.cluster)
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

    def _updateConfidence(self):
        # path = self.tile.pathTo(self.target_tile)
        # t_tcf = 0
        # for ti in range(1, len(path)):
        #     deltaCF = path[ti].getCF() - path[ti-1].getCF()
        #     t_tcf += deltaCF
        self.cf = 1 - abs(self.tile.getTCF() - self.target_tile.getTCF())


    # def kriging(self):
    #     tclass = self.target_tile.tclass
    #     tclass_tiles = getClassTiles(exclude=self.target_tile)

    def tiles_change_factor(self, target_tile):
        path = self.tile.pathTo(target_tile)
        total = 0
        for p_i in range(1, len(path)):
            total += path[p_i].getCF() - path[p_i-1].getCF()

        return total

    def makePredictions(self, target_sid, target_time, values, meas=None):
        self.target_tile = fetchTile_from_sid(target_sid)
        self._updateConfidence()
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

        self.prediction = total_pm1
        return total_pm1

    def makeCollabPrediction(self, cluster_predictions):
        cluster_bias = 1 - self.bias
        cluster_prediction = 0
        totalcf = 0

        for a in cluster_predictions:
            totalcf += cluster_predictions[a][1]

        for a in cluster_predictions:
            a_prediction = cluster_predictions[a][0]
            piece_of_bias = cluster_predictions[a][1]/totalcf
            cluster_prediction += piece_of_bias * a_prediction

        self.prediction = (self.prediction * self.bias) + (cluster_bias * cluster_prediction)
        return self.prediction

    def improveHeuristic(self, values, naive, collab, intervals):
        """
        improves bias distribution, model weights
        :param error: severity of error
        :return:
        """
        pass
        # if error <= 1:
        #     # increase agent bias
        #     self.bias = round(min(1, self.bias + 0.05), 3)
        # elif error > 1:
        #     self.bias = round(max(0.01, self.bias - 0.05), 3)


if __name__ == '__main__':
   pass
