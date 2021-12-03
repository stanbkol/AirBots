import numpy as np
from sklearn.metrics import mean_squared_error

from src.agents.ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate
from src.database.DbManager import Session
from src.database.Models import fetchTile_from_sid, Tile, getClassTiles


def _calc_error(x1, x2):
    return (x2 - x1) / x1


def _rel_diff(val, ref):
    return (val - ref) / abs(ref)


def _calc_squared_error(prediction, actual):
    return (actual - prediction) ** 2


class Agent(object):
    def __init__(self, sensor_id, thresholds, cluster_list, config=None, confidence=1):
        self.sid = sensor_id
        self._threshold = thresholds
        self.configs = config
        self.cf = confidence
        self.cluster = cluster_list
        self.models = self._initializeModels()
        self.error = 0
        self.n_error = 0
        self.p_error = 0
        self.target_tile = None
        self.tile = fetchTile_from_sid(self.sid)
        self.bias = 0
        self.prediction = 0
        self.integrity = 1
        self.data_integrity = 1

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
        """
        assigns weights to forecast models. models with lower error receive higher weights.
        :param predicts: dictionary of model_name: {pm: value}
        :param actual: actual value of pm
        :return: dictionary of model name: normalized weight
        """
        errors = {}
        total_se = 0
        for model_name in self._getModelNames():
            if model_name in predicts.keys():
                model_prediction = predicts[model_name]
                if model_prediction:
                    squared_error = _calc_squared_error(model_prediction['pm1'], actual)
                    total_se += squared_error
                    errors[model_name] = squared_error

        return {model_name: errors[model_name] / total_se for model_name in errors.keys()}

    def _updateConfidence(self):
        """
        updates the agent's confidence rating based on distance to target tile and difference in tile classification
        :return:
        """
        # path = self.tile.pathTo(self.target_tile)
        # t_tcf = 0
        # for ti in range(1, len(path)):
        #     deltaCF = path[ti].getCF() - path[ti-1].getCF()
        #     t_tcf += deltaCF

        tcf_delta = 1 - abs(self.tile.getTCF() - self.target_tile.getTCF())
        tile_dist_trust = self.getDistTrust()
        self.cf = np.mean([tcf_delta, tile_dist_trust])

    def tiles_change_factor(self, target_tile):
        """
        sums tile change factors in path from agent tile to prediction tile.
        :param target_tile: the target of prediction for agent
        :return: sum percent difference
        """
        path = self.tile.pathTo(target_tile)
        classified_path = [t for t in path if t.tclass is not None]
        total = 0
        for p_i in range(1, len(classified_path)):
            percent_diff = path[p_i].getCF() - path[p_i - 1].getCF()
            total += percent_diff

        return total

    def makePredictions(self, target_sid, target_time, values, meas=None):
        """
        makes predictions from all forecast models, inverse weights them according to error, and calculates agent model
        integrity and model data integrity average. returns a single predicted pm value.
        :param target_sid: sensor id of sensor residing in target tile
        :param target_time: datetime for which the prediction is made
        :param values: the pm values you are predicting
        :param meas: the measure object at target time (for validation)
        :return:
        """
        self.target_tile = fetchTile_from_sid(target_sid)
        self._updateConfidence()
        data_integrity = list()
        measure = meas
        predicts = {}
        for model in self._getModelNames():
            if model == 'mvr':
                prediction = self.models[model].makePrediction(target_time, values, target_sid=target_sid)
                if prediction:
                    data_integrity.append(float(self.models[model].db_imputed))
                    predicts[model] = prediction
            else:
                prediction = self.models[model].makePrediction(target_time, values)
                if prediction:
                    data_integrity.append(float(self.models[model].db_imputed))
                    predicts[model] = prediction

        # TODO: check for training flag to not reweight outside training, save weights on agent scope
        weights = self._weightModels(predicts, getattr(measure, 'pm1'))
        total_pm1 = 0
        for model in weights.keys():
            total_pm1 += weights[model] * predicts[model]['pm1']

        if total_pm1 == 0:
            return None

        self._integrity = round(len(predicts.keys()) / len(self._getModelNames()), 2)
        self._data_integrity = round(1-np.mean(data_integrity), 2)

        self.prediction = total_pm1
        return total_pm1

    def makeCollabPrediction(self, cluster_predictions):
        """
        makes a collaborative prediction with other agents in cluster
        :param cluster_predictions:
        :return: a single value for predicted pm
        """
        cluster_bias = 1 - self.bias
        cluster_prediction = 0
        totalcf = 0

        for a in cluster_predictions:
            totalcf += cluster_predictions[a][1]

        for a in cluster_predictions:
            a_prediction = cluster_predictions[a][0]
            piece_of_bias = cluster_predictions[a][1] / totalcf
            cluster_prediction += piece_of_bias * a_prediction

        self.prediction = (self.prediction * self.bias) + (cluster_bias * cluster_prediction)
        return self.prediction

    def getClusterPred(self, naive, collab):
        # print(f"\t\tn {naive}, type {type(naive)}")
        # print(f"\t\tc {collab}, type {type(collab)}")

        wnp = round((collab - (self.bias * naive)) / (1 - self.bias), 2)
        # print(f"\t\twnp: {wnp}, type:{type(wnp)}")
        return wnp

    def assessPerformance(self, values, naive, collab, intervals=None):
        """
        sets bias based on naive and cluster prediction performance
        :param values: actual values for hour intervals
        :param naive: agent naive predictions
        :param collab: collaboration predictions
        :param intervals: datetime hour intervals
        :return: adjusts agent bias by a percentage
        """
        naive_preds = naive[self.sid]
        cluster_preds = [self.getClusterPred(float(v), float(collab[i])) for i, v in enumerate(naive_preds)]

        cluster_mse = round(mean_squared_error(np.array(values), np.array(cluster_preds)), 2)
        naive_mse = round(mean_squared_error(np.array(values), np.array(naive_preds)), 2)

        fraction = cluster_mse / naive_mse
        rel_change = _rel_diff(cluster_mse, naive_mse)
        if fraction < 1:
            # print("naive has more error, decrease bias!")
            self.bias = max(0.51, round(self.bias - min([0.05, self.bias * (1 + rel_change)]), 2))
        elif fraction > 1:
            # print("collab has more error increase bias!")
            self.bias = min(0.95, round(self.bias + min([0.05, self.bias * (1 + rel_change)]), 2))

    def getDistTrust(self):
        """
        calculates the confidence factor based on tile distance (double-width coordinates) to target tile
        :return:
        """
        from src.map.HexGrid import DwHex, dw_distance, tile_dist_trust_factors
        start = DwHex(self.tile.x, self.tile.y)
        end = DwHex(self.target_tile.x, self.target_tile.y)
        dist = dw_distance(start, end)
        # print(f"dist {dist}")
        dist_factors = tile_dist_trust_factors()
        closest = min(range(1, len(dist_factors) + 1), key=lambda x: abs(x - dist))
        return dist_factors[closest - 1]


