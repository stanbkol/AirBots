from datetime import timedelta

from src.database.Models import *
from src.agents.Agents import *
from src.main.utils import *
from src.main.ExcelWriter import *


# Basic Average Model Aggregation
def avgAgg(preds):
    total = 0
    for k in preds:
        total += preds[k]
    return round(total / (len(preds)), 2)


def WeightedAggregation(preds, weights):
    """
    Method for aggregating dictionary of predictions based on dictionary of weights
    Used for aggregating by distance, error, and trust
    :param preds: dictionary of predictions, keys are agent.sid values
    :param weights: dictionary of weights, keys are agent.sid values
    :return: aggregated prediction
    """
    total = 0
    for s in preds:
        temp = preds[s]
        total += temp * weights.get(s)
    return total


def inverseWeights(data, pow=2):
    """
    calculates Inverse Distance Weights for known sensors relative to an unmeasured target.
    :param target: unmeasured target
    :param sensors: known points for which weights are made
    :param pow: determines the rate at which the weights decrease. Default is 2.
    :return: dict of sid-->weight on scale between 0 and 1.
    """
    sumsup = [(x[0], 1 / np.power(x[1], pow)) for x in data]
    suminf = sum(n for _, n in sumsup)
    weights = {x[0].sid: x[1] / suminf for x in sumsup}
    return weights


def mapAgentsToError(agents):
    """
    Helper function to prepare data for eventually use by inverseWeights function
    :param agents: dictionary of agent.sid to agent objects
    :return: returns list of tuples, (agent, agent.error)
    """
    data = []
    for a in agents:
        data.append((agents[a], agents[a].error))
    data.sort(key=lambda x: x[1])
    return data


def aggregatePrediction(preds, dist_weights, error_weights, tru_weights):
    """
    :param preds: dictionary of predictions mapped to agent.sid
    :param dist_weights: dictionary of weights based on inverse distance
    :param error_weights: dictionary of weights based on error
    :param tru_weights: dictionary of weights based on trust factor
    :return: dictionary of values, keys are the aggregation method, values are aggregated prediction
    """
    vals = {}
    avg = avgAgg(preds)
    dist = WeightedAggregation(preds, dist_weights)
    err = WeightedAggregation(preds, error_weights)
    # trust = trustAgg(preds, tru_weights)
    vals['average'] = round(avg, 2)
    vals['distance'] = round(dist, 2)
    vals['error'] = round(err, 2)
    return vals


def makeCluster(sid, sensors, n):
    """

    :param sid: sid of agent
    :param sensors: all active sensors in mesh
    :param n: size of cluster
    :return: list of agent.sids that represent the agents cluster
    """
    data = findNearestSensors(sid, sensors, n)
    cluster = []
    for sens, dist in data:
        cluster.append(sens.sid)
    return cluster


def getClusterError(cluster, agents):
    """

    :param cluster: list of sids that represent agents in the cluster
    :param agents: dictionary of agent objects mapped to agent.sid
    :return: returns the average error for all agents in the cluster
    """
    total = 0
    count = 0
    for a in cluster:
        total += agents[a].n_error
        count += 1
    return round(total / count, 2)


def targetInterval(time, interval):
    end = time + timedelta(hours=interval)
    start = time - timedelta(hours=interval)
    return start, end


class Central:
    agents = {}
    sensors = []
    agent_configs = {}
    thresholds = {}

    def __init__(self, model):
        self.data = getJson(model)
        self.writer = ExcelWriter(model + "_results.xlsx")
        self.error = {}
        self.model_params = self.data["model_params"]
        self.thresholds = self.data["thresholds"]
        self.sensors = self.data["sensors"]["on"]
        self.extractData()
        self.writer.initializeFile(self.agents)

    def extractData(self):
        self.thresholds = self.data["thresholds"]
        self.agent_configs = self.data["agent_configs"]
        for s in self.sensors:
            cluster = makeCluster(s, self.sensors, n=self.model_params['cluster_size'])
            a = Agent(s, self.thresholds, cluster, config=self.agent_configs)
            self.agents[a.sid] = a

    def sensorSummary(self, start, end):
        total = countInterval(start, end)
        for s in self.sensors:
            data = getMeasuresORM(s, start, end)
            print("Sensor :", s)
            if data:
                print("Data_Completeness %", round(((len(data)-1)/total), 2)*100)
            else:
                print("No Data for Training Interval")

    def getAllPredictions(self, target, time, val):
        """

        :param target: target sensor
        :param time: time for prediction
        :param val: real value for validation; to be used in forecastmodels
        :return: dictionary of predictions mapped to agent.sid
        """
        predictions = {}
        for a in self.agents:
            agent = self.agents[a]
            pred = agent.makePredictions(target, time, ["pm1"], meas=val)
            if pred:
                predictions[a] = (round(pred[0], 2), agent.cf)
        return predictions

    def makePrediction(self, target, time):
        """
        Main method that is responsible for making model predictions.
        Instantiates the training phase, and validates with a final prediction
        :param target: target for prediction
        :param time: time for prediction
        :return: N/A, displays/saves results to file
        """
        start, end = targetInterval(time, self.model_params["interval"])
        self.sensorSummary(start, end)
        # print("INITIAL PREDICTION")
        # self.finalPrediction(target, time)
        print("Training the Model")
        print("Target Sensor:", target)
        print("Interval between " + str(start) + " and " + str(end))
        self.trainModel(start, end, target)
        print("FINAL PREDICTION")
        self.finalPrediction(target, time)

    def trainModel(self, start_interval, end_interval, target):
        """

        :param start_interval: first interval for training
        :param end_interval: last interval for training
        :param target: target of prediction
        :return: N/A, no returns needed. controls flow of training the model
        """
        for i in range(1, self.model_params["num_iter"] + 1):
            cursor = start_interval
            collab_predictions = {sid: [] for sid in self.sensors}
            naive_predictions = {sid: [] for sid in self.sensors}
            values = []
            intervals = []
            end = end_interval + timedelta(hours=1)
            while cursor != end:
                print("Predictions for ", cursor)
                val = getMeasureORM(target, cursor)
                intervals.append(cursor)
                vals = {sid: [] for sid in self.sensors}
                if val:
                    values.append(val.pm1)
                    interval_preds = self.getAllPredictions(target, cursor, val)
                    for a in self.agents:
                        cluster_pred = {}
                        agent = self.agents[a]
                        for ca in agent.cluster:
                            cluster_pred[ca] = interval_preds[ca]
                        pred = round(agent.makeCollabPrediction(cluster_pred)[0], 2)
                        naive = interval_preds[a][0]
                        vals[a] = pred
                        collab_predictions[a].append(pred)
                        naive_predictions[a].append(naive)
                else:
                    print("No target validation data for-->", cursor)
                cursor += timedelta(hours=1)
            self.evaluateAgents(values, collab_predictions, naive_predictions)
            model_vals = self.aggregateModel(collab_predictions, countInterval(start_interval, end), target)
            self.evaluateModel(values, model_vals)
            self.writer.saveIter(values, collab_predictions, naive_predictions, model_vals, i, intervals, self.agents)
            self.writer.saveModel(i, self.agents, self.error)
            self.applyHeuristic(values, naive_predictions, collab_predictions, intervals)

    def evaluateAgents(self, values, predictions, naive_preds):
        """

        :param values: list of actual values for target sensor, for the training time interval
        :param predictions: dictionary of collaborative predictions mapped to agent.sid
        :param naive_preds: dictionary of naive predictions mapped to agent.sid
        :return: N/A, updates the naive/collab errors values in MAE and %, stored in the agent object
        """
        # print("values:")
        # print(values)
        for a in self.agents:
            agent = self.agents[a]
            # print("predictions for:", a)
            # print(predictions[a])
            agent.error = MAE(values, predictions[a])
            agent.n_error = MAE(values, naive_preds[a])
            agent.p_error = (p_err(values, naive_preds[a]), p_err(values, predictions[a]))

    def aggregateModel(self, preds, num_preds, target):
        """

        :param preds: dictionary of collab predictions mapped to each agent.sid, each value is list of predictions for training interval
        :param num_preds: size of training interval, for ease of iteration
        :param target: target sensor, to be used for distance weighting only
        :return: returns list of model aggregation values, in which case each entry in list is a dictionary, mapped to each kind of aggregation
        """
        model_vals = []
        dist_weights = inverseWeights(findNearestSensors(target, self.sensors))
        err_weights = inverseWeights(mapAgentsToError(self.agents))

        tru_weights = {}
        if num_preds > 1:
            for i in range(0, num_preds):
                interval_preds = {}
                for a in preds:
                    interval_preds[a] = preds[a][i]
                model_vals.append(aggregatePrediction(interval_preds, dist_weights, err_weights, tru_weights))
        else:
            model_vals.append(aggregatePrediction(preds, dist_weights, err_weights, tru_weights))
        return model_vals

    def evaluateModel(self, values, model_preds):
        """
        This method goes through the list of values from aggregate model, and evaluates each aggregation strategy
        :param values: real sensor values
        :param model_preds: list of dictionaries that show each kind of model aggregated prediction, for each interval
        :return: N/A, updates the model field to store the dictionary of error values, mapped to each aggregation method
        """
        error = {}
        avg_list = []
        dist_list = []
        err_list = []
        for e in model_preds:
            avg_list.append(e['average'])
            dist_list.append(e['distance'])
            err_list.append(e['error'])
        error['MAE'] = {'average': MAE(values, avg_list), 'dist_w': MAE(values, dist_list),
                        'error_w': MAE(values, err_list)}
        error['p'] = {'average': p_err(values, avg_list), 'dist_w': p_err(values, dist_list),
                      'error_w': p_err(values, err_list)}
        print(error)
        self.error = error

    def applyHeuristic(self, values, naive_predictions, collab_predictions, intervals):
        """
        This method takes all the predictions, and passes each agent only the necessary information they need to apply the heuristic
        Each agent gets a list of all naive predictions for himself and his cluster
        Each agent gets a list of all of his collaborative predictions for himself (used in forward checking)
        :param values: actual sensor values
        :param naive_predictions: full dictionary of all naive predictions for training interval
        :param collab_predictions: full dictionary of all collab predictions for training interval
        :param intervals: list of intervals
        :return: N/A, simply calls assessPerformance for each agent with necessary data
        """
        for a in self.agents:
            print("agent H:", a)
            agent = self.agents[a]
            key_list = [a]
            key_list.extend(agent.cluster)
            n_preds = {}
            for k in key_list:
                n_preds[k] = naive_predictions[k]
            # print("performance review")
            # print(f"\tactual: {values}")
            # print(f"\tnaive: {naive_predictions}")
            # print(f"\tcollabs: {collab_predictions[a]}")
            agent.assessPerformance(values, n_preds, collab_predictions[a], intervals)

    # TODO: update to include multiple prediction aggregation, rather than singular prediction
    # TODO: update to save results to excel file
    def finalPrediction(self, target, time):
        """
        This method makes the final prediction, based on the trained model
        :param target:
        :param time:
        :return: Will return the raw prediction+update results file
        """
        real_val = getMeasureORM(target, time)
        collab_preds = {}
        interval_preds = self.getAllPredictions(target, time, real_val)
        for a in self.agents:
            cluster_pred = {}
            agent = self.agents[a]
            for ca in agent.cluster:
                cluster_pred[ca] = interval_preds[ca]
            pred = round(agent.makeCollabPrediction(cluster_pred)[0], 2)
            collab_preds[a] = pred
        print("real val", real_val.pm1)
        model_val = self.aggregateModel(collab_preds, 1, target)[0]
        print("model pred", model_val)
        print("AVG-->AE:", round((real_val.pm1 - model_val['average']), 2))
        print("AVG-->%:", abs((model_val['average'] - real_val.pm1) / real_val.pm1))
        print("DIST-->AE:", round((real_val.pm1 - model_val['distance']), 2))
        print("DIST-->%:", abs((model_val['distance'] - real_val.pm1) / real_val.pm1))
        print("ERR-->AE:", round((real_val.pm1 - model_val['error']), 2))
        print("ERR-->%:", abs((model_val['error'] - real_val.pm1) / real_val.pm1))
