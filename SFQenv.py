from gym import Env
import numpy as np
from gym.spaces import Discrete, Box, Dict

max_sequence_length = 125


class SFQ(Env):
    def __init__(self):
        self.observation_space = Dict({
            "state": Box(low=np.zeros(125) * 3, high=np.zeros(125) * 3, dtype=int),
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
        return self.state, {}

if __name__=="__main__":
    env = SFQ()
    #print(env.state)
    for _ in range(max_sequence_length):
        action = env.action_space.sample()
        state,_,done,_=env.step(action)
        if done:
            print("Done!",state,env.index)

    print(env.reset())
