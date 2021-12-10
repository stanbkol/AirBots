import datetime
import itertools
import logging

from sklearn.metrics import mean_squared_error
import numpy as np

from src.agents.Agents import _rel_diff, Agent
from src.main.utils import MAE


logging.basicConfig(level=logging.DEBUG)


def mock_assPerformance(values, naive, collab):
    """
    test function for bias calculation

    """
    bias = 0.65
    naive_preds = naive

    def getClusteredPred(naive, colab, bias):
        wnp = round((colab - (bias * naive)) / (1 - bias), 2)
        print(f"\t\twnp: {wnp}")
        return wnp

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
    print(f"percent changed by: {1 + rel_change}")
    print(f"old bias: {bias}")

    print(f"bias diff: {bias * (1 + rel_change)}")
    if fraction < 1:
        print("naive has more error, decrease bias!")
        bias = round(bias - min([0.10, bias * (1 + rel_change)]), 2)
    elif fraction > 1:
        print("collab has more error increase bias!")
        bias = round(bias + min([0.10, bias * (1 + rel_change)]), 2)

    print(f"new bias: {bias}")


def test_bias():
    actual = [19.74, 20.88, 21.42, 22.21, 22.07, 24.02, 25.04, 25.01, 24.92, 24.2, 23.65, 23.63, 20.53, 19.84, 21.76,
              20.86, 20.94, 20.75, 21.62, 22.44, 24.07, 28.97, 28.89, 25.82]
    collab = [21.72, 17.61, 18.64, 26.74, 17.61, 20.64, 16.48, 10.69, 20.62, 18.48, 29.75, 31.63, 23.87, 29.57, 26.13,
              30.61, 27.87, 21.28, 34.61, 16.28, 12.34, 18.48, 22.29, 17.91]
    naive = [21.13, 17.5, 16.03, 32.26, 14.93, 22.88, 16.08, 6.77, 17.22, 15.08, 32.77, 35.22, 24.32, 30.55, 32.79,
             32.4, 35.52, 18.86, 38.34, 12.96, 8.36, 17.54, 22.53, 17.31]

    mock_assPerformance(actual, naive, collab)


def test_dist_tiles():
    from src.database.Models import getTileORM
    smith = Agent(11559, {}, [])
    smith.target_tile = getTileORM(29304)
    print(smith.getDistTrust())


def evaluate_config(model, config, actual, target_time, values, target_sid):
    num_p = 5

    actuals = [actual for _ in range(num_p)]
    if model.__class__.__name__ == 'MultiVariate':
        predictions = [model.makePrediction(target_time, values, target_sid=target_sid, config=config) for _ in range(num_p)]
    else:
        predictions = [model.makePrediction(target_time, values, config=config) for _ in range(num_p)]

    if predictions:
        logging.debug(f"eval config preds is 5: {len(predictions)==num_p}")
        return MAE(actuals, predictions)


def apply_forecast_heuristic(agent):
    """
    conducts a forward check search to find optimal configs for prediction models
    :return:
    """

    actual = 26.2
    target_time = datetime.datetime(2021, 5, 15, 11)
    values = ['pm1']
    target_sid = 11567

    base_configs = agent.configs.copy()
    base_configs.remove(base_configs['bias'])
    base_configs.remove(base_configs['sma']["interval_days"])
    base_configs.remove(base_configs['sma']["interval_hours"])
    base_configs.remove(base_configs['mvr']["interval_days"])
    logging.debug(f"initial configs: {base_configs}")
    # hooke-jeeves alg
    for k, v in base_configs.items():
        # params to test
        vals = v.items()
        fm = agent.models[k]

        origin_cfg = base_configs.copy()
        param_vector = list()
        for v in vals:
            # each x has a list of -,+ vals
            param_i = list()
            delta = 6 if 'hours' in v[0] else 1
            # calc points from - to +
            for d in range(-delta, delta+1, delta):
                if d != 0:
                    if delta == 1:
                        param_i.append( {max(1, v[1] + delta): v[0]})
                    else:
                        param_i.append( {max(12, v[1] + delta): v[0]})

            param_vector.append(param_i)

        permutations = list(itertools.product(*param_vector))
        logging.debug(f"forecast {k}")
        ratings = list()
        for p in permutations:
            test_cfg = origin_cfg.copy()
            for i in p:
                for g, s in i.items():
                    if s in test_cfg[k].keys():
                        test_cfg[k][s] = g
                        logging.debug(f"\tparam: {s}, v: {g}")
                logging.debug(f"configs: {test_cfg}")
                ratings.append(evaluate_config(fm, test_cfg, actual, target_time, values, target_sid))

        best_index = ratings.index(min(ratings))
        m_cfg = dict([(v,k) for x in permutations[best_index] for k, v in x.items()])
        direction = {k: m_cfg[k]-origin_cfg[k] for k in m_cfg}

        best_cfg = origin_cfg[k].copy()
        for d in direction:
            best_cfg[d] += direction[d]
        count = 0
        while True:
            test_cfg = best_cfg.copy()
            for d in direction:
                if 'hour' in d:
                    test_cfg[d] = max(12, test_cfg[d] + direction[d])
                else:
                    test_cfg[d] = max(1, test_cfg[d] + direction[d])

            mae = evaluate_config(fm, test_cfg, actual, target_time, values, target_sid)

            if count > 4 and mae > agent.error:
                break

            best_cfg = test_cfg

        for p in best_cfg:
            base_configs[k][p] = best_cfg[p]

        return base_configs




def test_prediction_heur():
    base_configs = {
      "bias":0.1,
      "nearby":{
         "n":3
      },
      "minmax":{
         "n":3
      },
      "sma":{
         "window":4,
         "interval_hours":48,
         "interval_days":0
      },
      "mvr":{
         "interval_hours":72,
         "interval_days":0
      }
    }
    thesholds = {
      "completeness":0.70,
      "trust":0.25,
      "confidence":0.60,
      "bias":0.10,
      "error":75
   }
    smith = Agent(5694, thresholds=thesholds, cluster_list=[11545, 11567, 5705], config=base_configs)

    best_configs = apply_forecast_heuristic(smith)


if __name__ == '__main__':
    # b = [{24: 'y'}, {72: 'y'}]
    # perms = [
    #     [{3:'x'}, {1:'x'}], [{24: 'y'}, {72: 'y'}]
    # ]
    # permutations = list(itertools.product(*perms))
    # print(permutations)
    # for p in permutations:
    #     for i in p:
    #         for k,v in i.items():
    #             print(v, k)
    bc = {
        "x": 2,
        "y": 48
    }

    t = ((1, 'a'), (2, 'b'))
    s = [({3: 'x'},)]
    z = dict([(v,k) for x in s[0] for k, v in x.items()])
    print(z)

    dir = [(k,z[k]-bc[k]) for k in z]
    print(dir)

    pass