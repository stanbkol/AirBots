# use this class to store the model configs,
# and individual agent configs (to be used in optimizing finding/storing best selfish configs)

class Config:
    def __init__(self, data):
        self.iterations = data["model_params"]["num_iter"]
        self.interval = data["model_params"]["interval"]
        self.cluster_size = data["model_params"]["cluster_size"]
        self.ratio = data["model_params"]["training_ratio"]
        self.thresholds = data["thresholds"]
        self.agent_configs = data["agent_configs"]
