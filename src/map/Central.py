from src.agents.AgentCluster import Cluster
from src.database.Models import *
from src.agents.Agents import *
from src.main.utils import *
from src.main.ExcelWriter import *
from src.agents import AgentCluster
import logging

from src.map.Config import Config


class Central:
    clusters = {}

    def __init__(self, model):
        logging.basicConfig(level=logging.INFO)
        self.model_file = model
        self.error = {}
        model_data = getJson(model)
        self.config = Config(model_data)
        self.sensors = model_data["sensors"]
        self.agent_results = {}
        self.model_results = {'error': 100, 'configs': {}}
        logging.info("Central System Created From " + str(self.model_file))

    def initializeModel(self, target):
        for s in self.sensors:
            cluster = Cluster(s, self.sensors, target, n=self.model_params['cluster_size'])
            a = Agent(s, self.thresholds, cluster, config=copy.deepcopy(self.config.agent_configs))
            a.updateConfidence(target)
            self.agents[a.sid] = a
            self.agent_results[a.sid] = {'error': 100, 'config': {}}
        logging.info("Agents Initialized:" + str(len(self.agents)))

    def sensorSummary(self, start, end):
        total = countInterval(start, end)
        logging.info("Sensors Before Check:" + str(len(self.sensors)))
        sensors_pass = []
        sensors_fail = []
        sensors_completeness = {}
        for s in self.sensors:
            data = getMeasuresORM(s, start, end)
            if data:
                data_completeness = round(((len(data)) / total), 2)
                if data_completeness > self.thresholds["completeness"]:
                    sensors_pass.append(s)
                    sensors_completeness[s] = data_completeness
                else:
                    sensors_fail.append(s)
            else:
                sensors_fail.append(s)
        logging.info("Sensors Passed:" + str(len(sensors_pass)))
        logging.info("Sensors Failed:" + str(len(sensors_fail)))
        self.sensors = sensors_pass

    def getAllPredictions(self, target, time, val, trainflag=True):
        """

        :param target: target sensor
        :param time: time for prediction
        :param val: real value for validation; to be used in forecastmodels
        :return: dictionary of predictions mapped to agent.sid
        """
        predictions = {}
        for a in self.agents:
            agent = self.agents[a]
            agent.training = trainflag
            pred = agent.makePredictions(target, time, [self.measure], meas=val)
            if pred:
                predictions[a] = (round(pred, 2), agent.cf)
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
                int(time.strftime('%Y%m%d%H'))) + "_" + self.measure + "_results.xlsx"

    def makePrediction(self, target, time, m, test=None):
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
        self.measure = m
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
        logging.info("Initialize Model Training:" + str(self.measure))
        logging.info("Target Sensor:" + str(target))
        logging.info("Interval between " + str(start) + " and " + str(end))
        self.trainModel(start, end, target)
        logging.info("Model Training Complete")
        real_val = getMeasureORM(target, time)
        # self.showResults()
        logging.info("Final Prediction for " + str(target) + ":" + str(
            int(time.strftime('%Y%m%d%H'))))
        logging.info("Real Value:" + str(getattr(real_val, self.measure)))
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
        training_intervals = int(round(self.ratio * total_intervals, 0))
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
                values.append(getattr(interval, self.measure))
                logging.info("Actual Value:" + str(getattr(interval, self.measure)))
                interval_preds = self.getAllPredictions(target, interval.date, interval)
                for a in self.agents:
                    cluster_pred = {}
                    agent = self.agents[a]
                    for ca in agent.cluster:
                        cluster_pred[ca] = interval_preds[ca]
                    pred = round(agent.makeCollabPrediction(cluster_pred), 2)
                    naive = interval_preds[a][0]
                    vals[a] = pred
                    collab_predictions[a].append(pred)
                    naive_predictions[a].append(naive)
                current_interval += 1
                if current_interval == training_intervals:
                    logging.info("Applying Agent Heuristics")
                    self.evaluateAgents(values, collab_predictions, naive_predictions)
                    model_vals = self.aggregateModel(collab_predictions, training_intervals)
                    self.evaluateModel(values, model_vals)
                    self.writer.saveIter(values, collab_predictions, naive_predictions, model_vals, i, intervals,
                                         self.agents, "Training")
                    self.applyHeuristic(values, naive_predictions, collab_predictions, intervals, target, iter=i)
                    collab_predictions = {sid: [] for sid in self.sensors}
                    naive_predictions = {sid: [] for sid in self.sensors}
                    values = []
                    intervals = []
            self.evaluateAgents(values, collab_predictions, naive_predictions)
            model_vals = self.aggregateModel(collab_predictions, validation_intervals)
            self.evaluateModel(values, model_vals)
            self.writer.saveIter(values, collab_predictions, naive_predictions, model_vals, i, intervals, self.agents, "Validation")
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
            logging.debug(f"eval naive preds: {naive_preds[a]}")
            logging.debug(f"eval collab preds: {predictions[a]}")
            naive_error = MAE(values, naive_preds[a])
            collab_error = MAE(values, predictions[a])
            percent_error = (p_err(values, naive_preds[a]), p_err(values, predictions[a]))
            agent.set_errors(collab_error, naive_error, percent_error)
            if agent.get_error() < self.agent_results[a]['error']:
                self.agent_results[a]['error'] = agent.get_error()
                self.agent_results[a]['config'] = copy.deepcopy(agent.configs)

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
            agent.assessPerformance(values, n_preds, collab_predictions[a], intervals, [self.measure], target_sid, iter)

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

        # single agent prediction
        smith = SingleAgent(target, self.sensors)
        single_agent_pred = smith.makePrediction(time, self.measure)
        logging.info(f"single agent pred: {single_agent_pred}")
        # ------
        for x in range(0, k):
            vals.append(getattr(real_val, self.measure))
            interval_preds = self.getAllPredictions(target, time, real_val, trainflag=False)
            # print(interval_preds)
            for a in self.agents:
                cluster_pred = {}
                agent = self.agents[a]
                for ca in agent.cluster:
                    cluster_pred[ca] = interval_preds[ca]
                pred = round(agent.makeCollabPrediction(cluster_pred), 2)
                naive = interval_preds[a][0]
                collab_preds[a].append(pred)
                naive_preds[a].append(naive)
        self.evaluateAgents(vals, collab_preds, naive_preds)
        model_vals = self.aggregateModel(collab_preds, 5)
        self.evaluateModel(vals, model_vals, False)
        self.writer.saveModel(i, self.agents, self.error, title)
        best_error = 100
        best_pred = 0
        for mv in model_vals:
            for v in mv:
                mae = abs(mv[v] - getattr(real_val, self.measure))
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



