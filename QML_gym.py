from typing import Any, SupportsFloat
from gymnasium import Env
import numpy as np
from itertools import permutations
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Discrete, Box
from qiskit import QuantumCircuit, QuantumRegister, BasicAer, transpile
from qiskit.quantum_info import state_fidelity


class QMLEnv(Env):
    def __init__(self, num_qubits, target_state):
        super().__init__()
        self.num_qubits = num_qubits
        self.target_state = target_state
        self.max_circuit_length = 14

        self.backend = BasicAer.get_backend('statevector_simulator') # TODO: fix code repeating
        self.register = QuantumRegister(self.num_qubits)
        self.qc = QuantumCircuit(self.register)

        self.single_qubit_operations = [self.qc.id, self.qc.x, self.qc.y,
                                        self.qc.z, self.qc.h, self.qc.s,
                                        self.qc.sdg, self.qc.t]
        self.two_qubit_operations = [self.qc.cx, self.qc.cy,
                                     self.qc.cz, self.qc.ch, self.qc.swap]
        self.three_qubit_operations = [self.qc.ccx, self.qc.cswap]
        self.rotational_gates = [self.qc.p, self.qc.rx,
                                 self.qc.ry, self.qc.rz]

        qubit_indices = [x for x in range(num_qubits)]
        self.two_qubit_perm, self.three_qubit_perm = [], []
        perm_two = permutations(qubit_indices, 2)
        for val in perm_two:
            self.two_qubit_perm.append(val)

        perm_three = permutations(qubit_indices, 3)
        for val in perm_three:
            self.three_qubit_perm.append(val)

        self.max_single_qubit_permutations = num_qubits
        self.max_two_qubit_permutations = len(self.two_qubit_perm)
        self.max_three_qubit_permutations = len(self.three_qubit_perm)

        self.single_qubit_threshold = len(self.single_qubit_operations) * self.max_single_qubit_permutations
        self.two_qubit_threshold = self.max_two_qubit_permutations * len(self.two_qubit_operations) + \
                                   self.single_qubit_threshold
        self.three_qubit_threshold = self.two_qubit_threshold + len(
        self.three_qubit_operations) * self.max_three_qubit_permutations

    def reset(
            self,
            *,
            seed=None,
            options=None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.backend = BasicAer.get_backend('statevector_simulator')
        self.register = QuantumRegister(self.num_qubits)
        self.qc = QuantumCircuit(self.register)
        self.index = 0
        return self._get_obs()

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = int(action)
        if action <= self.single_qubit_threshold:
            self.apply_single_qubit_gate(action)
        elif self.single_qubit_threshold < action <= self.two_qubit_threshold:
            self.apply_two_qubit_gate(action)
        elif self.two_qubit_threshold < action <= self.three_qubit_threshold:
            self.apply_three_qubit_gate(action)
        elif self.three_qubit_threshold < action:
            self.apply_rotational_gate(action)

        self._get_obs()
        done = False
        if self.index == self.max_circuit_length:
            done = True

        self.index += 1
        info = {'fidelity': self.fidelity}
        return self.state, self.fidelity, done, False, info

    def _get_obs(self):
        self.job = self.backend.run(transpile(self.qc, self.backend))
        self.state = self.job.result().get_statevector(self.qc)
        self.fidelity = state_fidelity(self.target_state, self.state)
        self.state = self.get_flattened_state()
        return self.state, {'fidelity': self.fidelity}

    def _get_state_str(self):
        pass

    def get_flattened_state(self):
        '''
        Reshapes complex-value multi-dimensional array into one-dimensional array of [Re(matrix),Im(matrix)]
        :param matrix: input matrix
        :return: 1D float array
        '''
        return np.concatenate((self.state.real.flatten(), self.state.imag.flatten()))

    def apply_single_qubit_gate(self, action):
        action_type = action // self.num_qubits  # decode aciton number into gate type and qubit index
        qubit_index = action % self.num_qubits
        action_type = self.single_qubit_operations[action_type]  # quantum gate
        action_type(qubit_index)  # apply gate

    def apply_two_qubit_gate(self, action):

        action = action - self.single_qubit_threshold
        action_type = action // self.max_two_qubit_permutations
        action = action - action_type * self.max_two_qubit_permutations
        first_qubit_index, second_qubit_index = self.two_qubit_perm[action]

        action_type = self.two_qubit_operations[action_type]
        action_type(first_qubit_index, second_qubit_index)

    def apply_three_qubit_gate(self, action):

        action = action - self.two_qubit_threshold
        action_type = action // self.max_three_qubit_permutations
        action = action - action_type * self.max_three_qubit_permutations
        first_qubit_index, second_qubit_index, third_qubit_index = self.three_qubit_perm[action]
        action_type = self.three_qubit_operations[action_type]
        action_type(first_qubit_index, second_qubit_index, third_qubit_index)

    def apply_rotational_gate(self, action):

        if len(np.where(self.state == action)) > 0:
            idx = int(np.where(self.state == action)[0][0])
        else:
            idx = int(np.where(self.state == action)[0][1])

        action = action - self.three_qubit_threshold
        rotational_action_type = action // self.num_qubits
        qubit_index = action % self.num_qubits

        # angle = angles[idx]
        # action_type = rotational_gates_[rotational_action_type]
        # action_type(angle, qubit_index)


if __name__ == "__main__":
    num_qubits = 5
    target_backend = BasicAer.get_backend('statevector_simulator')
    target_register = QuantumRegister(num_qubits)
    target_qc = QuantumCircuit(target_register)
    target_qc.cnot(0, 2)
    job = target_backend.run(transpile(target_qc, target_backend))
    target_state = job.result().get_statevector(target_qc)


    env = QMLEnv(num_qubits=5, target_state=target_state)

    print(env.reset())
