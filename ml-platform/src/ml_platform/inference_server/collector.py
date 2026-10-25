from ml_platform.inference_server.config import CollectorConfig


class Collector:
    def __init__(self, config: CollectorConfig):

        self.config = config
        self.max_observations = config.max_observations
        self.sample_index = 0
        self.observations = []


    def collect(self, obs):
        obs.sample_index = self.sample_index

        if len(self.observations) >= self.max_observations:
            self.observations = self.observations[:10]

        self.observations.append(obs)
        self.sample_index += 1

    def get_n_latest_observations(self, n):
        n = min(n, len(self.observations))
        return self.observations[-n: ]

    def get_observations(self, start_index, end_index=-1):
        start_index = max(start_index, 0)

        if end_index >= len(self.observations):
            end_index = -1

        return self.observations[start_index: end_index]

    def reset(self):
        self.observations = []
        self.sample_index = 0