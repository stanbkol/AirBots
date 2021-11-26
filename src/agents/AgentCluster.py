

class Cluster(object):
    def __init__(self, agents, relationships):
        self.agents = agents
        self.edges = relationships

    def collab_prediction(self):
        return NotImplemented()

    def agent_heuristic(self):
        return NotImplemented()

