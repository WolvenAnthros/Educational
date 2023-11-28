import pprint
from itertools import permutations
import numpy as np
from qiskit import BasicAer
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import state_fidelity
# from scipy.misc import derivative
# from copy import deepcopy
from geneticalgorithm import geneticalgorithm as ga

backend = BasicAer.get_backend('statevector_simulator')
num_qubits = 10  # число кубитов в системе

target_q = QuantumRegister(num_qubits)
target_qc = QuantumCircuit(target_q)

target_qc.sdg(1)
target_qc.t(4)
target_qc.cnot(4, 9)
target_qc.ch(2, 6)
target_qc.z(0)
target_qc.s(3)
target_qc.ccx(3, 1, 7)
target_qc.cswap(7, 8, 2)
target_qc.x(7)
target_qc.y(5)
target_qc.t(7)
target_qc.z(6)
target_qc.ccz(0, 8, 2)
target_qc.cnot(5, 9)

job = backend.run(transpile(target_qc, backend))
target_state = job.result().get_statevector(target_qc)
pprint.pprint(target_state)
pprint.pprint(target_state.real)
print(target_qc.draw(output='text'))

# INITIAL_STATE = []
# INITIAL_STATE = tuple(INITIAL_STATE)
# INITIAL_INDEX = 0
# sequence_length = args['pulse_array_length']
sequence_length = 14
total_length = sequence_length * 2  # +angles
gate_dtype = np.array([['int']] * sequence_length)
rotation_dtype = np.array([['real']] * sequence_length)
vartype = np.concatenate((gate_dtype, rotation_dtype))
gate_range = np.array([[0, 2009]] * sequence_length)
angle_range = np.array([[0, 2 * np.pi]] * sequence_length)
varbound = np.concatenate((gate_range, angle_range))

qubit_indices = [x for x in range(num_qubits)]
# TODO: optimize code
max_single_qubit_permutations = num_qubits

two_qubit_perm, three_qubit_perm = [], []
perm_two = permutations(qubit_indices, 2)
for val in perm_two:
    two_qubit_perm.append(val)

perm_three = permutations(qubit_indices, 3)
for val in perm_three:
    three_qubit_perm.append(val)

max_two_qubit_permutations = len(two_qubit_perm)
max_three_qubit_permutations = len(three_qubit_perm)
print(
    f'Action Space: {max_three_qubit_permutations * 2 + max_two_qubit_permutations * 5 + max_single_qubit_permutations * 8}')


def move(state):  # , idx,action
    """
    Набор из всех невзаимозаменямых гейтов, выраженный в виде 285 действий
    :param state: квантовое состояние
    :return: infidelity
    """

    # state = list(state)  # decode

    q = QuantumRegister(num_qubits)

    qc = QuantumCircuit(q)

    # applied_action = action
    angles = state[sequence_length:]
    state = state[:sequence_length]

    singe_qubit_operations_ = [qc.id, qc.x, qc.y, qc.z, qc.h, qc.s, qc.sdg, qc.t]
    two_qubit_operations_ = [qc.cx, qc.cy, qc.cz, qc.ch, qc.swap]
    three_qubit_operations_ = [qc.ccx, qc.cswap]

    rotational_gates_ = [qc.p, qc.rx, qc.ry, qc.rz]  # qc.u,

    two_qubit_threshold = max_two_qubit_permutations * len(two_qubit_operations_) + \
                          max_single_qubit_permutations * len(singe_qubit_operations_)

    three_qubit_threshold = two_qubit_threshold + len(three_qubit_operations_) * max_three_qubit_permutations

    # state.append(applied_action)
    for action in state:
        action = int(action)
        if action < max_single_qubit_permutations * len(singe_qubit_operations_):  # single qubit operation
            action_type = action // num_qubits
            qubit_index = action % num_qubits

            action_type = singe_qubit_operations_[action_type]  # quantum gate
            action_type(qubit_index)

        elif two_qubit_threshold > \
                action >= max_single_qubit_permutations * len(singe_qubit_operations_):  # two-qubit operation
            action = action - num_qubits * 8
            action_type = action // max_two_qubit_permutations  # REMIND: check this formula for different number of qubits
            action = action - action_type * max_two_qubit_permutations
            first_qubit_index, second_qubit_index = two_qubit_perm[action]

            action_type = two_qubit_operations_[action_type]
            action_type(first_qubit_index, second_qubit_index)

        elif three_qubit_threshold > action >= two_qubit_threshold:
            action = action - two_qubit_threshold
            action_type = action // max_three_qubit_permutations
            action = action - action_type * max_three_qubit_permutations
            first_qubit_index, second_qubit_index, third_qubit_index = three_qubit_perm[action]

            action_type = three_qubit_operations_[action_type]
            action_type(first_qubit_index, second_qubit_index, third_qubit_index)
        elif action >= three_qubit_threshold:
            if len(np.where(state == action)) > 0:
                idx = int(np.where(state == action)[0][0])
            else:
                idx = int(np.where(state == action)[0][1])

            action = action - three_qubit_threshold
            rotational_action_type = action // num_qubits
            qubit_index = action % num_qubits
            # desired_angle = adam(optimize_rotational_gate

            angle = angles[idx]
            action_type = rotational_gates_[rotational_action_type]
            action_type(angle, qubit_index)

    job = backend.run(transpile(qc, backend))
    qc_state = job.result().get_statevector(qc)
    pprint.pprint(qc_state)
    reward = state_fidelity(target_state, qc_state)
    print(qc.draw(output='text'))
    print(f'Infidelity: {1 - reward:.2e}')

    # done = False
    # if idx == game_length - 2:
    #     done = True

    # state = tuple(state)
    return 1 - reward  # state, reward, done


# def adam(func):
#     point = 2
#     m = 0
#     s = 0
#     beta_1 = 0.9
#     beta_2 = 0.999
#     learning_rate = 0.05  # 0.01
#     for t in range(1, 1000):
#         d_func = derivative(func, point, dx=1e-4)
#         m = beta_1 * m - (1 - beta_1) * d_func
#         s = beta_2 * s + (1 - beta_2) * d_func * d_func
#         m_hat = m / (1 - beta_1 ** t)
#         s_hat = s / (1 - beta_2 ** t)
#         point = point + learning_rate * m_hat / np.sqrt(s_hat + 1e-6)
#     return point

# def optimize_rotational_gate(angle):
#     q_ = deepcopy(q)
#     qc_ = deepcopy(qc)
#     rotational_gates = [qc_.p, qc_.rx, qc_.ry, qc_.rz]  # qc_.u,
#     action_type_ = rotational_gates[rotational_action_type]
#     action_type_(angle, qubit_index)
#     job = backend.run(transpile(qc_, backend))
#     qc_state = job.result().get_statevector(qc_)
#     reward = state_fidelity(target_state, qc_state)
#     return 1 - reward

if __name__ == "__main__":
    result_state = np.array( [1.15300000e+03, 1.72400000e+03, 1.04000000e+02, 9.94000000e+02,
     2.00600000e+03, 2.80000000e+01, 1.85700000e+03 ,1.24400000e+03,
     2.49000000e+02, 5.29000000e+02 ,5.79000000e+02 ,1.96300000e+03,
     1.48000000e+02 ,1.16400000e+03 ,2.87230836e+00 ,2.99308382e+00,
     1.02925651e+00 ,1.74378943e+00 ,3.91890888e-01 ,2.65557658e+00,
     6.00388063e+00 ,5.92007098e+00 ,5.18184980e+00 ,3.22405344e+00,
     1.22377120e+00 ,3.76954528e+00 ,4.85310119e+00 ,5.74370273e+00])
    reward = move(result_state)


    # algorithm_param = {'max_num_iteration': 300, 'population_size': 1000, \
    #                    'mutation_probability': 0.1, \
    #                    'elit_ratio': 0.01, \
    #                    'crossover_probability': 0.5, \
    #                    'parents_portion': 0.3, \
    #                    'crossover_type': 'uniform', \
    #                    'max_iteration_without_improv': None}
    #
    # #
    # model = ga(function=move, dimension=total_length, variable_type_mixed=vartype, variable_boundaries=varbound,
    #            algorithm_parameters=algorithm_param)
    #
    # model.run()


