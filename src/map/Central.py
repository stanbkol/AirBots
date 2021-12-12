from datetime import timedelta
from os import walk
import os
from src.database.Models import *
from src.agents.Agents import *
from src.main.utils import *
from src.main.ExcelWriter import *
import logging


def tileSummary():
    tiles = getTilesORM(1)
    tc = {}
    for t in tiles:
        if t.tclass in tc.keys():
            tc[t.tclass] += 1
        else:
            tc[t.tclass] = 1
    return tc


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


def aggregatePrediction(preds, error_weights, tru_weights):
    """
    :param preds: dictionary of predictions mapped to agent.sid
    :param dist_weights: dictionary of weights based on inverse distance
    :param error_weights: dictionary of weights based on error
    :param tru_weights: dictionary of weights based on trust factor
    :return: dictionary of values, keys are the aggregation method, values are aggregated prediction
    """
    vals = {}
    avg = avgAgg(preds)
    err = WeightedAggregation(preds, error_weights)
    trust = WeightedAggregation(preds, tru_weights)
    vals['average'] = round(avg, 2)
    vals['error'] = round(err, 2)
    vals['trust'] = round(trust, 2)
    return vals


def makeCluster(sid, sensors, target_sid, n):
    """

    :param sid: sid of agent
    :param sensors: all active sensors in mesh
    :param n: size of cluster
    :return: list of agent.sids that represent the agents cluster
    """
    with Session as sesh:
        tid = sesh.query(Sensor.tid).where(Sensor.sid == target_sid).first()[0]
        if tid:
            tile = getTileORM(tid)
    tclass_sensors = sameTClassSids(tile.tclass, sensors)
    data = findNearestSensors(sid, tclass_sensors, n)
    cluster = []
    for sens, dist in data:
        cluster.append(sens.sid)
    # checkCluster(target_sid, cluster, sid)
    return cluster


def checkCluster(target, cluster, sid):
    targ = fetchTile_from_sid(target)
    print("Cluster For A:", sid)
    print("Target Class:"+targ.road+targ.tclass)
    for a in cluster:
        temp = fetchTile_from_sid(a)
        print("Agent #"+str(a)+"-->" + temp.road + temp.tclass)


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
    return time - timedelta(hours=interval), time


def getTrustWeights(agents, sensors_completeness):
    trust_factors = {}
    total_tf = 0
    for a in agents:
        agent = agents[a]
        dc = round(sensors_completeness[a], 2)
        trust_factors[a] = round(dc * agent.cf * agent.integrity(), 2)
        total_tf += trust_factors[a]
    #     print("Trust Summary for S:", a)
    #     print("CF:", agent.cf)
    #     print("Calculated Trust Value-->", trust_factors[a])
    trust_weights = {}
    for a in trust_factors:
        trust_weights[a] = round(trust_factors[a]/total_tf, 2)
    # print("TF-->", trust_factors)
    # print("TW-->", trust_weights)
    return trust_weights


def archiveResults(docs):
    filenames = next(walk(docs), (None, None, []))[2]
    for file in filenames:
        temp = file.split('_')
        if temp[-1] == 'results.xlsx':
            old_file_path = docs+"\\"+file
            new_file_path = docs+"\\results\\"+file
            os.replace(old_file_path, new_file_path)


class Central:
    agents = {}
    sensors = []
    agent_configs = {}
    thresholds = {}

    def __init__(self, model):
        logging.basicConfig(level=logging.INFO)
        self.data = getJson(model)
        self.model_file = model
        self.error = {}
        self.agent_results = {}
        self.model_results = {'error': 100, 'configs': {}}
        self.model_params = self.data["model_params"]
        self.iterations = self.model_params["num_iter"]
        self.interval = self.model_params["interval"]
        self.cluster_size = self.model_params["cluster_size"]
        self.ratio = self.model_params["training_ratio"]
        self.thresholds = self.data["thresholds"]
        logging.info("Central System Created From " + str(self.model_file))

    def extractData(self, target):
        self.agent_configs = self.data["agent_configs"]
        for s in self.sensors:
            cluster = makeCluster(s, self.sensors, target, n=self.model_params['cluster_size'])
            a = Agent(s, self.thresholds, cluster, config=copy.deepcopy(self.agent_configs))
            a.updateConfidence(target)
            self.agents[a.sid] = a
            self.agent_results[a.sid] = {'error': 100, 'config': {}}
        logging.info("Agents Initialized:" + str(len(self.agents)))

    def sensorSummary(self, start, end):
        total = countInterval(start, end)
        logging.info("Sensors Before Check:" + str(len(self.sensors)))
        sensors_pass = []
        sensors_fail = []
        self.sensors_completeness = {}
        for s in self.sensors:
            data = getMeasuresORM(s, start, end)
            if data:
                data_completeness = round(((len(data)) / total), 2)
                if data_completeness > self.thresholds["completeness"]:
                    sensors_pass.append(s)
                    self.sensors_completeness[s] = data_completeness
                else:
                    sensors_fail.append(s)
            else:
                sensors_fail.append(s)
        logging.info("Sensors Passed:" + str(len(sensors_pass)))
        logging.info("Sensors Failed:" + str(len(sensors_fail)))
        self.sensors = sensors_pass

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
            else:
                predictions[a] = pred
        return predictions

    def formatResults(self, target, time, test):
        if test:
            test_string = "_" + test + "_"
            if test == "iterations":
                test_string += str(self.iterations)
            if test == "intervals":
                test_string += str(self.interval)
            if test == "cluster":
                test_string += str(self.cluster_size)
            if test == "ratio":
                test_string += str(self.ratio)
            return self.model_file + "_" + str(target) + "_" + str(
                int(time.strftime('%Y%m%d%H'))) + "_" + test_string + "_results.xlsx"
        else:
            return self.model_file + "_" + str(target) + "_" + str(
                int(time.strftime('%Y%m%d%H'))) + "_results.xlsx"

    def makePrediction(self, target, time, test=None):
        """
        Main method that is responsible for making model predictions.
        Instantiates the training phase, and validates with a final prediction
        :param test:
        :param target: target sid for prediction
        :param time: time for prediction
        :return: N/A, displays/saves results to file
        """
        start, end = targetInterval(time, self.interval)
        self.sensors = self.data["sensors"]["on"]
        results_file = self.formatResults(target, time, test)
        print(results_file)
        self.writer = ExcelWriter(results_file)
        if target in self.sensors:
            self.sensors.remove(target)
        self.sensorSummary(start, end)
        logging.info("Initializing Agents")
        target_tile = fetchTile_from_sid(target)
        self.extractData(target)
        self.writer.initializeFile(self.agents)
        logging.info("Initialize Model Training")
        logging.info("Target Sensor:" + str(target))
        logging.info("Interval between " + str(start) + " and " + str(end))
        self.trainModel(start, end, target)
        logging.info("Model Training Complete")
        real_val = getMeasureORM(target, time)
        # self.showResults()
        logging.info("Final Prediction for " + str(target) + ":" + str(
            int(time.strftime('%Y%m%d%H'))))
        logging.info("Real Value:" + str(real_val.pm1))
        logging.info("Historical Agent Mesh-->Final Prediction")
        self.updateModel()
        historical_best = self.finalPrediction(target, time, 5, real_val, self.model_params["num_iter"] + 1, "H")
        logging.info("Prediction:" + str(historical_best[0]))
        logging.info("Error:" + str(historical_best[1]))
        logging.info("Selfish Agent Mesh-->Final Prediction")
        self.updateAgents()
        selfish_best = self.finalPrediction(target, time, 5, real_val, self.model_params["num_iter"] + 2, "S")
        logging.info("Prediction:" + str(selfish_best[0]))
        logging.info("Error:" + str(selfish_best[1]))
        self.writer.saveAgentConfigs(self.agent_results)
        self.writer.saveModelBest(self.model_results['configs'])

    def trainModel(self, start_interval, end_interval, target):
        """

        :param start_interval: first interval for training
        :param end_interval: last interval for training
        :param target: target sid of prediction
        :return: N/A, no returns needed. controls flow of training the model
        """
        orm_values = getMeasuresORM(target, start_interval, end_interval)
        total_intervals = len(orm_values)
        training_intervals = round(self.ratio * total_intervals, 0)
        validation_intervals = int(total_intervals-training_intervals)
        logging.info("Interval Breakdown--> T:" + str(training_intervals) + " V:" + str(validation_intervals))
        for i in range(1, self.iterations + 1):
            logging.info("Beginning Iteration #" + str(i))
            current_interval = 0
            collab_predictions = {sid: [] for sid in self.sensors}
            naive_predictions = {sid: [] for sid in self.sensors}
            values = []
            intervals = []
            for interval in orm_values:
                logging.info("Predictions for " + str(interval.date))
                intervals.append(interval.date)
                vals = {sid: [] for sid in self.sensors}
                values.append(interval.pm1)
                interval_preds = self.getAllPredictions(target, interval.date, interval)
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
                current_interval += 1
                if current_interval == training_intervals:
                    logging.info("Applying Agent Heuristics")
                    self.evaluateAgents(values, collab_predictions, naive_predictions)
                    self.applyHeuristic(values, naive_predictions, collab_predictions, intervals, target, iter=i)
                    collab_predictions = {sid: [] for sid in self.sensors}
                    naive_predictions = {sid: [] for sid in self.sensors}
                    values = []
                    intervals = []
            self.evaluateAgents(values, collab_predictions, naive_predictions)
            model_vals = self.aggregateModel(collab_predictions, validation_intervals, target)
            self.evaluateModel(values, model_vals)
            self.writer.saveIter(values, collab_predictions, naive_predictions, model_vals, i, intervals, self.agents)
            self.writer.saveModel(i, self.agents, self.error, "I")

    def evaluateAgents(self, values, predictions, naive_preds):
        """

        :param values: list of actual values for target sensor, for the training time interval
        :param predictions: dictionary of collaborative predictions mapped to agent.sid
        :param naive_preds: dictionary of naive predictions mapped to agent.sid
        :return: N/A, updates the naive/collab errors values in MAE and %, stored in the agent object
        """
        for a in self.agents:
            agent = self.agents[a]
            agent.error = MAE(values, predictions[a])
            agent.n_error = MAE(values, naive_preds[a])
            agent.p_error = (p_err(values, naive_preds[a]), p_err(values, predictions[a]))
            if agent.error < self.agent_results[a]['error']:
                self.agent_results[a]['error'] = agent.error
                self.agent_results[a]['config'] = copy.deepcopy(agent.configs)

    def aggregateModel(self, preds, num_preds, target):
        """

        :param preds: dictionary of collab predictions mapped to each agent.sid, each value is list of predictions for training interval
        :param num_preds: size of training interval, for ease of iteration
        :param target: target sensor, to be used for distance weighting only
        :return: returns list of model aggregation values, in which case each entry in list is a dictionary, mapped to each kind of aggregation
        """
        model_vals = []
        err_weights = inverseWeights(mapAgentsToError(self.agents))
        tru_weights = getTrustWeights(self.agents, self.sensors_completeness)
        if num_preds > 1:
            for i in range(0, num_preds):
                interval_preds = {}
                for a in preds:
                    interval_preds[a] = preds[a][i]
                model_vals.append(aggregatePrediction(interval_preds, err_weights, tru_weights))
        else:
            model_vals.append(aggregatePrediction(preds, err_weights, tru_weights))
        return model_vals

    def evaluateModel(self, values, model_preds, update_historical=True):
        """
        This method goes through the list of values from aggregate model, and evaluates each aggregation strategy
        :param values: real sensor values
        :param model_preds: list of dictionaries that show each kind of model aggregated prediction, for each interval
        :return: N/A, updates the model field to store the dictionary of error values, mapped to each aggregation method
        """
        error = {}
        avg_list = []
        tf_list = []
        err_list = []
        for e in model_preds:
            avg_list.append(e['average'])
            tf_list.append(e['trust'])
            err_list.append(e['error'])
        error['MAE'] = {'average': MAE(values, avg_list), 'trust_w': MAE(values, tf_list),
                        'error_w': MAE(values, err_list)}
        error['p'] = {'average': p_err(values, avg_list), 'trust_w': p_err(values, tf_list),
                      'error_w': p_err(values, err_list)}
        lowest_error = 100
        for m in error['MAE']:
            current_error = error['MAE'][m]
            if current_error < lowest_error:
                lowest_error = current_error

        if update_historical:
            if lowest_error < self.model_results['error']:
                logging.info("new historical best mesh")
                self.model_results['error'] = lowest_error
                self.model_results['configs'] = self.getMeshConfig()
        self.error = error

    def applyHeuristic(self, values, naive_predictions, collab_predictions, intervals, target_sid, iter):
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

            agent = self.agents[a]
            key_list = [a]
            key_list.extend(agent.cluster)
            n_preds = {}
            for k in key_list:
                n_preds[k] = naive_predictions[k]
            agent.assessPerformance(values, n_preds, collab_predictions[a], intervals, ["pm1"], target_sid, iter)

    # TODO: update to include multiple prediction aggregation, rather than singular prediction
    # TODO: update to save results to excel file
    def finalPrediction(self, target, time, k, real_val, i, title):
        """
        This method makes the final prediction, based on the trained model
        :param target:
        :param time:
        :return: Will return the raw prediction+update results file
        """
        logging.info("Start Final Prediction")
        vals = []
        collab_preds = {sid: [] for sid in self.sensors}
        naive_preds = {sid: [] for sid in self.sensors}
        for x in range(0, k):
            vals.append(real_val.pm1)
            interval_preds = self.getAllPredictions(target, time, real_val)
            # print(interval_preds)
            for a in self.agents:
                cluster_pred = {}
                agent = self.agents[a]
                for ca in agent.cluster:
                    cluster_pred[ca] = interval_preds[ca]
                pred = round(agent.makeCollabPrediction(cluster_pred)[0], 2)
                naive = interval_preds[a][0]
                collab_preds[a].append(pred)
                naive_preds[a].append(naive)
        self.evaluateAgents(vals, collab_preds, naive_preds)
        model_vals = self.aggregateModel(collab_preds, 5, target)
        self.evaluateModel(vals, model_vals, False)
        self.writer.saveModel(i, self.agents, self.error, title)
        best_error = 100
        best_pred = 0
        for mv in model_vals:
            for v in mv:
                mae = abs(mv[v] - real_val.pm1)
                if mae < best_error:
                    best_error = mae
                    best_pred = mv[v]
        return best_pred, best_error

    def updateAgents(self):
        logging.info("Updating Agent Configs")
        for a in self.agents:
            agent = self.agents[a]
            agent.configs = self.agent_results[a]['config']
        logging.info("Agent Update Complete.")

    def updateModel(self):
        logging.info("Updating Model Configs")
        for a in self.agents:
            agent = self.agents[a]
            agent.configs = self.model_results['configs'][a]
        logging.info("Model Update Complete.")

    def getMeshConfig(self):
        mesh_config = {}
        for a in self.agents:
            agent = self.agents[a]
            mesh_config[a] = copy.deepcopy(agent.configs)
        return mesh_config

    def showResults(self):
        print("Best Agent Configs")
        for a in self.agent_results:
            print(self.agent_results[a]['config'])
        print("Best Model Config")
        for a in self.model_results['configs']:
            print(self.model_results['configs'][a])



