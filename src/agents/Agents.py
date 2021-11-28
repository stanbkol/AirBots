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
    return (actual-prediction)**2


class Agent(object):
    def __init__(self, sensor_id, thresholds, cluster_list, config=None, confidence=1):
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
            if model_name in predicts.keys():
                model_prediction = predicts[model_name]
                if model_prediction:
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

        tcf_delta = 1 - abs(self.tile.getTCF() - self.target_tile.getTCF())
        tile_dist_trust = self.getDistTrust()
        self.cf = np.mean([tcf_delta, tile_dist_trust])

    def tiles_change_factor(self, target_tile):
        path = self.tile.pathTo(target_tile)
        classified_path = [t for t in path if t.tclass is not None]
        total = 0
        for p_i in range(1, len(classified_path)):
            percent_diff = path[p_i].getCF() - path[p_i-1].getCF()
            total += percent_diff

        return total

    def makePredictions(self, target_sid, target_time, values, meas=None):
        self.target_tile = fetchTile_from_sid(target_sid)
        self._updateConfidence()
        measure = meas
        predicts = {}
        for model in self._getModelNames():
            if model == 'mvr':
                prediction = self.models[model].makePrediction(target_time, values, target_sid=target_sid)
                if prediction:
                    predicts[model] = prediction
            else:
                prediction = self.models[model].makePrediction(target_time, values)
                if prediction:
                    predicts[model] = prediction

        weights = self._weightModels(predicts, getattr(measure, 'pm1'))
        total_pm1 = 0
        for model in weights.keys():
            total_pm1 += weights[model] * predicts[model]['pm1']

        if total_pm1 == 0:
            return None

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

    def getClusterPred(self, naive, collab):
        return round((collab - (self.bias * naive)) / (1-self.bias), 2)

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
        cluster_preds = [self.getClusterPred(v, collab[i]) for i,v in enumerate(naive_preds)]
        print(naive_preds)
        print(cluster_preds)

        cluster_mse = round(mean_squared_error(np.array(values), np.array(cluster_preds)), 2)
        naive_mse = round(mean_squared_error(np.array(values), np.array(naive_preds)), 2)
        fraction = cluster_mse / naive_mse
        rel_change = _rel_diff(cluster_mse, naive_mse)

        if fraction < 1:
            # print("naive has more error, decrease bias!")
            self.bias = round(self.bias - min([0.30, self.bias * (1 + rel_change)]), 2)
        elif fraction > 1:
            # print("collab has more error increase bias!")
            self.bias = round(self.bias + min([0.30, self.bias * (1 + rel_change)]), 2)

    def getDistTrust(self):
        from src.map.HexGrid import DwHex, dw_distance, tile_dist_trust_factors
        start = DwHex(self.tile.x, self.tile.y)
        end = DwHex(self.target_tile.x, self.target_tile.y)
        dist = dw_distance(start, end)
        # print(f"dist {dist}")
        dist_factors = tile_dist_trust_factors()
        closest = min(range(1, len(dist_factors) + 1), key=lambda x: abs(x - dist))
        return dist_factors[closest - 1]


def test_dist_tiles():
    from src.database.Models import getTileORM
    smith = Agent(11559, {}, [])
    smith.target_tile = getTileORM(29304)
    print(smith.getDistTrust())


def mock_assPerformance(values, naive, collab):
    bias = 0.65
    naive_preds = naive

    def getClusteredPred(naive, colab, bias):
        return round((colab - (bias * naive)) / (1 - bias),2)

    cluster_preds = [getClusteredPred(v, collab[i], bias) for i, v in enumerate(naive_preds)]

    # for i,v in enumerate(naive_preds):
    #     cp = getClusteredPred(v, collab[i], 0.7)
    #     print(f"collab: {collab[i]}")
    #     print(f"np:{v}")
    #     print(f"cp: {cp}")
    #     print(f"collabPrediction: {bias*v + 0.3*cp}")

    cluster_mse = round(mean_squared_error(np.array(values), np.array(cluster_preds)), 2)
    naive_mse = round(mean_squared_error(np.array(values), np.array(naive_preds)), 2)

    print(f"naive_preds: {naive_preds}")
    print(f"cluster_preds: {cluster_preds}")

    print(f"naive_mse: {naive_mse}")
    print(f"cluster_mse: {cluster_mse}")

    # if c_mse > nmse shift bias to self, else give more weight to cluster.

    fraction = cluster_mse / naive_mse
    rel_change = _rel_diff(cluster_mse, naive_mse)

    print(f"cluster_ms/naive_mse: {fraction}")
    print(f"rel_change(collab, rel=naive): {rel_change}")
    print(f"percent changed by: {1+rel_change}")
    print(f"old bias: {bias}")

    print(f"bias diff: {bias * (1+rel_change)}")
    if fraction < 1:
        print("naive has more error, decrease bias!")
        bias = round(bias - min([0.10, bias * (1+rel_change)]),2)
    elif fraction > 1:
        print("collab has more error increase bias!")
        bias = round(bias + min([0.10, bias * (1+rel_change)]),2)

    print(f"new bias: {bias}")


def test_bias():
    actual = [19.74,20.88,21.42,22.21,22.07,24.02,25.04,25.01,24.92,24.2,23.65,23.63,20.53,19.84,21.76,20.86,20.94,20.75,21.62,22.44,24.07,28.97,28.89,25.82]
    collab = [21.72,17.61,18.64,26.74,17.61,20.64,16.48,10.69,20.62,18.48,29.75,31.63,23.87,29.57,26.13,30.61,27.87,21.28,34.61,16.28,12.34,18.48,22.29,17.91]
    naive = [21.13,17.5,16.03,32.26,14.93,22.88,16.08,6.77,17.22,15.08,32.77,35.22,24.32,30.55,32.79,32.4,35.52,18.86,38.34,12.96,8.36,17.54,22.53,17.31]

    actual2 = [13.0, 9.56, 8.57, 6.93]
    naive2 = [15.22, 16.13, 9.86, 13.87]
    collab2 = [1.08, 7.3, 16.53, 26.0]
    mock_assPerformance(actual2, naive2, collab2)


if __name__ == '__main__':
    # test_dist_tiles()
    test_bias()
