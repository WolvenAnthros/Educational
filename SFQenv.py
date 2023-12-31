from gymnasium import Env
import numpy as np
from gymnasium.spaces import Discrete, Box
from SFQ_calc import reward_calculation, identity_matrix, dimensions, u_t_plus, u_t_minus, u_t_zero

max_sequence_length = 125


class SFQ(Env):
    def __init__(self):
        self.observation_space = Box(low=np.zeros((dimensions * dimensions) * 2),
                                     high=np.ones((dimensions * dimensions) * 2), dtype=np.float32)

        self.action_space = Discrete(3)

        # self.initial_state = np.zeros(max_sequence_length, dtype=int)pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        self.initial_state = identity_matrix
        self.state = self.initial_state.copy()
        self.index = 0
        self.fidelity = 0
        self.reward_old = 0
        self.seed = None
        # self.action_history = []

    def _get_obs(self):
        return np.concatenate((self.state.real.flatten(), self.state.imag.flatten()))

    def step(self, action):
        # self.state[self.index] = action - 1

        pulse = action - 1

        if pulse == 1:
            self.state = self.state @ u_t_plus
        elif pulse == -1:
            self.state = self.state @ u_t_minus
        elif pulse == 0:
            self.state = self.state @ u_t_zero

        self.index += 1

        self.fidelity = reward_calculation(self.state)
        log_reward = (self.fidelity - np.log(1-self.fidelity)) ** 2
        reward = log_reward - self.reward_old

        done = False
        if self.index >= max_sequence_length:
            done = True

        info = {'fidelity': self.fidelity}

        self.reward_old = log_reward

        return self.reshape_state(), reward, done, False, info

    def _get_state_str(self):
        state_str = ''
        for pulse in self.state:
            state_str += str(pulse)
        return state_str

    def reshape_state(self):
        '''
        Reshapes complex-value multi-dimensional array into one-dimensional array of [Re(matrix),Im(matrix)]
        :return: 1D float array
        '''
        # assert np.shape(self.state)[0] == dimensions
        return np.concatenate((self.state.real.flatten(), self.state.imag.flatten()))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        self.index = 0
        self.fidelity = 0
        self.reward_old = 0
        return self.reshape_state(), {}


if __name__ == "__main__":
    env = SFQ()
    # print(env.state)
    for _ in range(max_sequence_length):
        action = env.action_space.sample()
        state, fidelity, done, *_ = env.step(action)
        if done:
            print("Done!", state, env.index)
            print(f'Pulse list: {env.action_history}, \n Fidelity: {fidelity:.2f}')

    print(env.reset()[0])
    # print(reward_calculation(
    #     pulse_list=np.array([1, 1, 1, 0, 1, 1, 0, 0, -1, 0, 1, -1, -1, 1, 0, 1, 1, 0, 0, 0, -1, 0, -1, -1,
    #                          -1, 1, 1, 0, 1, 1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 0, -1, -1, 1, 0, 0, 0, 1, -1,
    #                          1, 1, 0, 0, -1, -1, 1, -1, 1, -1, 0, -1, -1, 0, 0, 1, -1, -1, 0, 1, 1, 1, 1, -1,
    #                          1, -1, -1, 0, 1, 1, 1, 0, 0, 0, 1, -1, -1, 0, 0, 1, -1, 1, 0, 1, 0, 0, 1, -1,
    #                          0, 1, -1, 1, 0, -1, -1, 1, -1, -1, -1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, -1,
    #                          1, 1, 1, -1, 1, ])))
