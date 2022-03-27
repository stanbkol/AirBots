from src.database.DbManager import Session
from src.database.Models import Sensor, getTileORM, sameTClassSids, findNearestSensors

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
        total += agents[a].get_n_error()
        count += 1
    return round(total / count, 2)


class Cluster(object):
    def __init__(self, agent, sensors, target, size):
        self.agent = agent
        self.sensors = sensors
        self.target = target
        self.size = size

    def makeCluster(self):
        with Session as sesh:
            tid = sesh.query(Sensor.tid).where(Sensor.sid == self.target).first()[0]
            if tid:
                tile = getTileORM(tid)
        tclass_sensors = sameTClassSids(tile.tclass, self.sensors)
        data = findNearestSensors(self.agent, tclass_sensors, self.size)
        cluster = []
        for sens, dist in data:
            cluster.append(sens.sid)
        # checkCluster(target_sid, cluster, sid)
        return cluster

    def collab_prediction(self):
        return NotImplemented()

    def agent_heuristic(self):
        return NotImplemented()

