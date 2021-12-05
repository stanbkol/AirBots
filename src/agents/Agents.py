import numpy as np
from sklearn.metrics import mean_squared_error

from src.agents.ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate
from src.database.DbManager import Session
from src.database.Models import fetchTile_from_sid, Tile, getClassTiles
import logging


def _calc_error(x1, x2):
    return (x2 - x1) / x1


def _rel_diff(val, ref):
    return (val - ref) / abs(ref)


def _calc_squared_error(prediction, actual):
    return (actual - prediction) ** 2


def getModelNames():
    return ['rand', 'nearby', 'minmax', 'sma', 'mvr']


logging.basicConfig(level=logging.INFO)


class Agent(object):
    training = True
    predictions = {model:None for model in getModelNames()}

    def __init__(self, sensor_id, thresholds, cluster_list, config=None, confidence=1):
        self.sid = sensor_id
        self._threshold = thresholds
        self.configs = config
        self.cf = confidence
        self.cluster = cluster_list
        self.completeness = 0
        self.models = self._initializeModels()
        self.error = 0
        self.n_error = 0
        self.p_error = 0
        self.target_tile = None
        self.tile = fetchTile_from_sid(self.sid)
        self.bias_thresh = thresholds['bias']
        self.prediction = 0
        self._integrity = 1
        self._data_integrity = 1
        # initially all models have equal weights
        self.model_weights = {name: 1/len(getModelNames()) for name in getModelNames()}

    def integrity(self):
        return round(self._integrity * self._data_integrity, 2)

    def _initializeModels(self):
        nearby = self.configs['nearby']
        minmax = self.configs['minmax']
        sma = self.configs['sma']
        mvr = self.configs['mvr']

        models = {'rand': RandomModel(self.sid, self.cluster),
                    'nearby': NearbyAverage(self.sid, self.cluster, config=nearby),
                    'minmax': MinMaxModel(self.sid, self.cluster, config=minmax),
                    'sma': CmaModel(self.sid, self.cluster, config=sma),
                    'mvr': MultiVariate(self.sid, self.cluster, config=mvr)
                }

        return models

    def _weightModels(self, predicts, actual):
        """
        assigns weights to forecast models. models with lower error receive higher weights.
        :param predicts: dictionary of model_name: {pm: value}
        :param actual: actual value of pm
        :return: dictionary of model name: normalized weight
        """
        errors = {}
        total_se = 0
        for model_name in getModelNames():
            if model_name in predicts.keys():
                model_prediction = predicts[model_name]
                if model_prediction:
                    squared_error = _calc_squared_error(model_prediction['pm1'], actual)
                    total_se += squared_error
                    errors[model_name] = squared_error

        return {model_name: errors[model_name] / total_se for model_name in errors.keys()}

    def updateConfidence(self, target_sid):
        """
        updates the agent's confidence rating based on distance to target tile and difference in tile classification
        :return:
        """
        self.target_tile = fetchTile_from_sid(target_sid)
        tcf_delta = 1 - abs(self.tile.getTCF() - self.target_tile.getTCF())
        tile_dist_trust = self.getDistTrust()
        self.cf = round(np.mean([tcf_delta, tile_dist_trust]), 2)

        logging.debug(f"agent dist_trust: {tile_dist_trust}")
        logging.debug(f"agent tcf_delta: {tcf_delta}")
        logging.debug(f"agent cf: {self.cf}")

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
        data_integrity = list()
        measure = meas
        predicts = {}
        for model in getModelNames():
            if model == 'mvr':
                prediction = self.models[model].makePrediction(target_time, values, target_sid=target_sid)
                if prediction:
                    data_integrity.append(float(self.models[model]._imputed))
                    predicts[model] = prediction
            else:
                prediction = self.models[model].makePrediction(target_time, values)
                if prediction:
                    data_integrity.append(float(self.models[model]._imputed))
                    predicts[model] = prediction

        # TODO: check for training flag to not reweight outside training, save weights on agent scope
        if self.training:
            self.model_weights = self._weightModels(predicts, getattr(measure, 'pm1'))
        total_pm1 = 0
        for model in self.model_weights.keys():
            total_pm1 += self.model_weights[model] * predicts[model]['pm1']

        if total_pm1 == 0:
            return None

        self._integrity = round(len(predicts.keys()) / len(getModelNames()), 2)
        self._data_integrity = round(1-np.mean(data_integrity), 2)
        logging.debug(f"agent integrity: {self._integrity}")
        logging.debug(f"agent data_integrity: {self._data_integrity}")

        self.prediction = total_pm1
        logging.debug(f"prediction: {self.prediction}")
        return total_pm1

    def makeCollabPrediction(self, cluster_predictions, configs_state=None):
        """
        makes a collaborative prediction with other agents in cluster
        :param cluster_predictions:
        :return: a single value for predicted pm
        """
        cluster_bias = 1 - self.configs['bias']
        cluster_prediction = 0
        totalcf = 0

        for a in cluster_predictions:
            totalcf += cluster_predictions[a][1]

        for a in cluster_predictions:
            a_prediction = cluster_predictions[a][0]
            piece_of_bias = cluster_predictions[a][1] / totalcf
            cluster_prediction += piece_of_bias * a_prediction

        self.prediction = (self.prediction * self.configs['bias']) + (cluster_bias * cluster_prediction)
        return self.prediction

    def getClusterPred(self, naive, collab):
        # print(f"\t\tn {naive}, type {type(naive)}")
        # print(f"\t\tc {collab}, type {type(collab)}")

        wnp = round((collab - (self.configs['bias'] * naive)) / (1 - self.configs['bias']), 2)
        # print(f"\t\twnp: {wnp}, type:{type(wnp)}")
        return wnp

    def _within_bias_threshold(self, value):
        if -self.bias_thresh <= value <= self.bias_thresh and round(value, 2) == value:
            return True
        return False

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
        logging.debug(f"cmse/nmse: {fraction}")
        logging.debug(f"rel_change: {rel_change}")

        # print("Asessing for S:", self.sid)
        # print("pre bias 2:", self.configs['bias'])
        # if not between -0.1 and 0.1, then apply bias change.
        if not self._within_bias_threshold(rel_change):
            if fraction < 1:
                # print("naive has more error, decrease bias!")
                self.configs['bias'] = max(0.51, round(self.configs['bias'] - min([0.05, self.configs['bias'] * (1 + rel_change)]), 2))
            elif fraction > 1:
                # print("collab has more error increase bias!")
                self.configs['bias'] = min(0.95, round(self.configs['bias'] + min([0.05, self.configs['bias'] * (1 + rel_change)]), 2))
        # print("post bias 2:", self.configs['bias'])

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
        dist_factors = tile_dist_trust_factors(50)
        closest = min(range(1, len(dist_factors) + 1), key=lambda x: abs(x - dist))
        return dist_factors[closest - 1]


