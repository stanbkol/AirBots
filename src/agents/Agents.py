import copy
import itertools
import sys

import numpy as np
from sklearn.metrics import mean_squared_error

from src.agents.ForecastModels import RandomModel, NearbyAverage, MinMaxModel, CmaModel, MultiVariate
from src.database.DbManager import Session
from src.database.Models import fetchTile_from_sid, Tile, getClassTiles, getSidFromTile
import logging

from src.main.utils import MAE


def _calc_error(x1, x2):
    return (x2 - x1) / x1


def _rel_diff(val, ref):
    return (val - ref) / abs(ref)


def _calc_squared_error(prediction, actual):
    return (actual - prediction) ** 2


def getModelNames():
    return ['rand', 'nearby', 'minmax', 'sma', 'mvr']


class Agent(object):
    logging.basicConfig(level=logging.INFO)

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
        self.errors = []
        self.target_tile = None
        self.tile = fetchTile_from_sid(self.sid)
        self.bias_thresh = thresholds['bias']
        self.prediction = 0
        self._integrity = 1
        self._data_integrity = 1
        # initially all models have equal weights
        self.model_weights = []
        logging.debug(f"Agent {self.sid}'s cluster: {cluster_list}")

    def integrity(self):
        return round(self._integrity * self._data_integrity, 2)

    def get_error(self):
        return self.errors[-1]['collab']

    def get_n_error(self):
        return self.errors[-1]['naive']

    def get_np_error(self):
        return self.errors[-1]['percent'][0]

    def get_cp_error(self):
        return self.errors[-1]['percent'][1]

    def set_errors(self, collab, naive, percent):
        self.errors.append({
            'collab': collab,
            'naive': naive,
            'percent': percent
        })

    def fc_weights(self):
        return self.model_weights[-1]

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

    def _weightModels(self, predicts, actual, pm_str):
        """
        assigns weights to forecast models. models with lower error receive higher weights.
        :param predicts: dictionary of {model_name: {pm: value}}
        :param actual: actual value of pm
        :return: dictionary of model name: normalized weight
        """
        errors = {}
        total_se = 0
        for model_name in getModelNames():
            if model_name in predicts.keys():
                model_predictions = predicts[model_name]
                if model_predictions:
                    squared_error = _calc_squared_error(model_predictions[pm_str], actual)
                    total_se += squared_error
                    errors.update({model_name: squared_error})

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

        # logging.debug(f"agent dist_trust: {tile_dist_trust}")
        # logging.debug(f"agent tcf_delta: {tcf_delta}")
        # logging.debug(f"agent cf: {self.cf}")

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
        if not values:
            logging.error("no values selected for prediction")
            return ValueError

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

        # update weights of fc models for each pm prediction
        if self.training:
            pm_model_weights = {}
            for pm in values:
                pm_model_weights.update({pm: self._weightModels(predicts, getattr(measure, pm), pm)})

            self.model_weights.append(pm_model_weights)

        # aggregate predictions of pms with weights
        results = {}
        for pm in values:
            totalpm = 0
            logging.debug(f"fc_weights: {self.fc_weights()[pm]}")
            for model in self.fc_weights()[pm].keys():
                # logging.info(f"weight: {self.fc_weights()[pm][model]}, predV: {predicts[model][pm]}")
                totalpm += self.fc_weights()[pm][model] * predicts[model][pm]

            results.update({pm: totalpm})

        if not results:
            return None

        # logging.debug(f"pred res: {results}")
        self._integrity = round(len(predicts.keys()) / len(getModelNames()), 2)
        self._data_integrity = round(1-np.mean(data_integrity), 2)
        logging.debug(f"agent integrity: {self._integrity}")
        logging.debug(f"agent data_integrity: {self._data_integrity}")

        # Assumes prediction for only 1 pm value.
        # logging.debug(f"prediction_value: {results[values[0]]}")
        self.prediction = results[values[0]]
        logging.info(f"agent {self.sid}, prediction: {self.prediction}, cf: {self.cf}")
        return self.prediction

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

    def evaluate_config(self, model, config, actuals, target_times, values, target_sid):

        if model.__class__.__name__ == 'MultiVariate':
            predictions = [model.makePrediction(target_time, values, target_sid=target_sid, new_config=config)
                           for target_time in target_times]
        else:
            predictions = [model.makePrediction(target_time, values, new_config=config) for target_time in target_times]

        if predictions:
            pred_res = [x[values[0]] for x in predictions]
            # logging.debug(f"eval config preds is: {len(predictions)}")
            logging.debug(f"eval_config---pred_res: {pred_res}")
            return MAE(actuals, pred_res)

    def apply_forecast_heuristic(self, actuals, target_times, values, target_sid):
        """
        conducts a direct search algorithm to find optimal configs for prediction models
        :return:
        """
        base_configs = copy.deepcopy(self.configs)
        logging.debug(f"base_configs: {base_configs}")
        del base_configs['bias']
        del base_configs['sma']["interval_days"]
        # del base_configs['sma']["interval_hours"]
        del base_configs['mvr']["interval_days"]
        # logging.debug(f"initial configs: {base_configs}")
        # logging.debug(f"agent configs: {self.configs}")
        logging.debug(f"base cfg keys: {base_configs.keys()}")
        # hooke-jeeves alg
        for k, v in base_configs.items():
            logging.debug(f"INSPECTING {k} CFGS")
            # params to test
            vals = v.items()
            fm = self.models[k]

            origin_cfg = copy.deepcopy(base_configs[k])
            param_vector = list()
            for v in vals:
                logging.debug(f"{k} value : {v}")
                # each x has a list of -,+ vals
                param_i = list()
                if 'hours' in v[0]:
                    delta = 6
                    min_val = 12
                elif 'window' in v[0]:
                    delta = 1
                    min_val = 2
                else:
                    delta = 1
                    min_val = 1
                # calc points from - to +
                for d in range(-delta, delta + 1, delta):
                    if d != 0:
                        param_i.append({max(min_val, v[1] + d): v[0]})

                param_vector.append(param_i)
                # logging.debug(f"{k}, param v{ param_vector}")
            permutations = list(itertools.product(*param_vector))
            ratings = list()
            logging.debug(f"{k} move space: {permutations}")
            logging.debug(f"DETERMINING BEST DIRECTION")
            for p in permutations:
                test_cfg = copy.deepcopy(origin_cfg)
                for i in p:
                    for g, s in i.items():
                        if s in test_cfg.keys():
                            test_cfg[s] = g
                logging.debug(f"EVALUATING CONFIG: {test_cfg}")
                ratings.append(self.evaluate_config(fm, test_cfg, actuals, target_times, values, target_sid))

            # logging.debug(f"direction ratings: {ratings}")
            best_index = ratings.index(min(ratings))
            # logging.debug(f"best index: {best_index}")
            m_cfg = dict([(v, k) for x in permutations[best_index] for k, v in x.items()])
            direction = {c: m_cfg[c] - origin_cfg[c] for c in m_cfg}
            # update best_cfg with best direction
            best_cfg = copy.deepcopy(origin_cfg)
            for d in direction:
                best_cfg[d] += direction[d]

            # logging.debug(f"best_cfg init: {best_cfg}")
            count = 0
            best_mae = self.get_n_error()
            # logging.debug(f"{k} vector going in direction: {direction}")

            while True:
                logging.debug(f"\tCOUNT: {count}")
                test_cfg = copy.deepcopy(best_cfg)
                for d in direction:
                    if 'hours' in d:
                        min_val = 12
                    elif 'window' in d:
                        min_val = 2
                    else:
                        min_val = 1

                    test_cfg[d] = max(min_val, test_cfg[d] + direction[d])

                mae = self.evaluate_config(fm, test_cfg, actuals, target_times, values, target_sid)

                logging.debug(f"\tbest_mae: {best_mae}--> cfg_mae: {mae}")
                if mae > best_mae or count == 5:
                    break

                if mae < best_mae:
                    best_mae = mae
                    best_cfg = copy.deepcopy(test_cfg)

                count += 1

            for p in best_cfg:
                base_configs[k][p] = best_cfg[p]

        return base_configs

    def assessPerformance(self, actuals, naive, collab, target_times, values, t_sid, iter):
        """
        sets bias based on naive and cluster prediction performance
        :param actuals: actual values for hour intervals
        :param naive: agent naive predictions
        :param collab: collaboration predictions
        :param intervals: datetime hour intervals
        :return: adjusts agent bias by a percentage
        """
        logging.debug(f"ITERATION: {iter}, AGENT: {self.sid}")

        naive_preds = naive[self.sid]
        cluster_preds = [self.getClusterPred(float(v), float(collab[i])) for i, v in enumerate(naive_preds)]

        cluster_mse = round(mean_squared_error(np.array(actuals), np.array(cluster_preds)), 2)
        naive_mse = round(mean_squared_error(np.array(actuals), np.array(naive_preds)), 2)

        fraction = cluster_mse / naive_mse
        rel_change = _rel_diff(cluster_mse, naive_mse)
        # logging.debug(f"cmse/nmse: {fraction}")
        # logging.debug(f"rel_change: {rel_change}")

        if not self._within_bias_threshold(rel_change):
            if fraction < 1:
                # print("naive has more error, decrease bias!")
                self.configs['bias'] = max(0.51, round(self.configs['bias'] - min([0.05, self.configs['bias'] * (1 + rel_change)]), 2))
            elif fraction > 1:
                # print("collab has more error increase bias!")
                self.configs['bias'] = min(0.95, round(self.configs['bias'] + min([0.05, self.configs['bias'] * (1 + rel_change)]), 2))
        # print("post bias 2:", self.configs['bias'])

        logging.debug(f"AGENT: {self.sid}, INIT CONFIGS: {self.configs}")
        best_configs = self.apply_forecast_heuristic(actuals, target_times, values, t_sid)
        for k,v in best_configs.items():
            for i,w in v.items():
                self.configs[k][i] = best_configs[k][i]
        logging.debug(f"AGENT: {self.sid}, IMP CONFIGS: {self.configs}")

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


