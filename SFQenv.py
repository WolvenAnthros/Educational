from gym import Env
import numpy as np
from gym.spaces import Discrete, Box, Dict

max_sequence_length = 125


class SFQ(Env):
    def __init__(self):
        self.observation_space = Dict({
            "state": Box(low=np.ones(125) * 3, high=np.ones(125) * 3, dtype=int),
            "index": Box(low=np.array([0]), high=np.array([max_sequence_length]),dtype=int) })

        self.action_space = Discrete(3)

        self.state = self.observation_space["state"].sample()
        self.index = 0 # self.observation_space["index"]


    def step(self, action):
        self.state[self.index] = action -1

        self.index += 1

        reward = None

        done = False
        if self.index >= max_sequence_length:
            done = True

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = self.observation_space["state"].sample()
        self.index = 0


env = SFQ()
#print(env.state)
for _ in range(max_sequence_length):
    action = env.action_space.sample()
    env.step(action)
print(env.state)
print(env.index)

example = Discrete(3)
