
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
        data.append((agents[a], agents[a].get_error()))
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

    def aggregateModel(self, preds, num_preds):
        """

        :param preds: dictionary of collab predictions mapped to each agent.sid, each value is list of predictions for training interval
        :param num_preds: size of training interval, for ease of iteration
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

class Aggregator:

    def __init__(self):
        self.predictions = {}

    def setPredictions(self, preds):
        self.predictions = preds

    # Basic Average Model Aggregation
    def avgAgg(self, preds):
        total = 0
        for k in preds:
            total += preds[k]
        return round(total / (len(preds)), 2)

