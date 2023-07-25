import abc

class Agent:
    def __init__(self):
        return

    def train(self, state, action, reward, next_state, done):
        pass

    def save(self, path):
        pass

    def act(self, observation):
        pass

    # static method to load the model
    @staticmethod
    def load(path):
        pass

