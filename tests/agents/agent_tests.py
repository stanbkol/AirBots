from sklearn.metrics import mean_squared_error
import numpy as np

from src.agents.Agents import _rel_diff, Agent


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


if __name__ == '__main__':
    pass