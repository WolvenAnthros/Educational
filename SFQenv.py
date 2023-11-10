from gymnasium import Env
import numpy as np
from gymnasium.spaces import Discrete, Box, Dict
from Educational.SFQ_calc import reward_calculation

max_sequence_length = 125


class SFQ(Env):
    def __init__(self):
        self.observation_space = Box(low=np.ones(max_sequence_length, dtype=int) * -1,
                                     high=np.ones(max_sequence_length, dtype=int) * 1, dtype=int)

        self.action_space = Discrete(3)

        # self.state = self.observation_space["state"].sample()
        # self.state = self.observation_space.sample()
        self. initial_state = np.zeros(max_sequence_length, dtype=int)
        self.state = self.initial_state.copy()
        self.index = 0  # self.observation_space["index"]
        self.fidelity = 0

    def _get_obs(self):
        return self.state

    def step(self, action):
        self.state[self.index] = action - 1

        self.index += 1


        #self.true_fidelity = reward_calculation(self.state)

        done = False
        if self.index >= max_sequence_length:
            self.fidelity = reward_calculation(self.state) * 100 - 60
            done = True
        info = {}

        return self.state, self.fidelity, done, False, info

    def _get_state_str(self):
        state_str = ''
        for pulse in self.state:
            state_str += str(pulse)
        return state_str

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.state = self.observation_space["state"].sample()
        self.state = self.initial_state.copy()
        self.index = 0
        self.fidelity = 0
        return self.state, {}


if __name__ == "__main__":
    env = SFQ()
    # print(env.state)
    for _ in range(max_sequence_length):
        action = env.action_space.sample()
        state, *_, done, _ = env.step(action)
        if done:
            print("Done!", state, env.index)

    print(env.reset())
    print(reward_calculation(
        pulse_list=np.array([1, 1, 1, 0, 1, 1, 0, 0, -1, 0, 1, -1, -1, 1, 0, 1, 1, 0, 0, 0, -1, 0, -1, -1,
                             -1, 1, 1, 0, 1, 1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 1, 0, 0, 0, 1, -1,
                             1, 1, 0, 0, -1, -1, 1, -1, 1, -1, 0, -1, -1, 0, 0, 1, -1, -1, 0, 1, 1, 1, 1, -1,
                             1, -1, -1, 0, 1, 1, 1, 0, 0, 0, 1, -1, -1, 0, 0, 1, -1, 1, 0, 1, 0, 0, 1, -1,
                             0, 1, -1, 1, 0, -1, -1, 1, -1, -1, -1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1,
                             1, 1, 1, -1, 1, ])))
