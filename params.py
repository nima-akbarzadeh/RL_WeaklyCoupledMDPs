import numpy as np


# Define the cost function for each arm
class CostReward:

    def __init__(self, num_arms: int, num_states: int, function_type: str, rccc_wrt_max=1.5,
                 constant=0.5):
        self.num_a = num_arms
        self.num_s = num_states
        self.costs = rccc_wrt_max * np.ones([self.num_s, 2, self.num_a])
        if function_type == 'constant':
            self.costs[:, 0, :] = constant * np.ones([self.num_s, 1, self.num_a])
        elif function_type == 'linear':
            for a in range(self.num_a):
                self.costs[:, 0, a] = np.linspace(0, self.num_s-1, num=self.num_s)/(self.num_s - 1)
        elif function_type == 'quadratic':
            for a in range(self.num_a):
                for s in range(self.num_s):
                    self.costs[s, 0, a] = s ** 2 / (self.num_s - 1) ** 2
        else:
            print('The function type is UNDEFINED!')
        self.costs = self.costs / np.max(self.costs)
        self.rewards = 1.0 - self.costs


# Define the Markov dynamics for each arm
class MarkovDynamics:

    def __init__(self, num_arms: int, num_states: int, prob_remain, transition_type):
        self.num_s = num_states
        self.num_a = num_arms
        self.transitions = self.purereset_and_deteriorate(prob_remain, transition_type)

    def purereset_and_deteriorate(self, prob_remain, transition_type):
        transitions = np.zeros([self.num_s, self.num_s, 2, self.num_a])
        for a in range(self.num_a):
            for s in range(self.num_s):
                transitions[s, :, 1, a] = \
                    np.concatenate([np.ones([1, 1]), np.zeros([1, self.num_s - 1])], 1)
            if transition_type == 1:
                for s in range(self.num_s - 1):
                    transitions[s, s, 0, a] = prob_remain[a]
                    transitions[s, s+1, 0, a] = 1 - prob_remain[a]
                transitions[self.num_s-1, self.num_s-1, 0, a] = 1
            elif transition_type == 2:
                for s in range(self.num_s - 2):
                    transitions[s, s, 0, a] = prob_remain[a]
                    transitions[s, s+1, 0, a] = (1 - prob_remain[a]) / 2
                    transitions[s, s+2, 0, a] = (1 - prob_remain[a]) / 2
                transitions[self.num_s-2, self.num_s-2, 0, a] = prob_remain[a]
                transitions[self.num_s-2, self.num_s-1, 0, a] = 1 - prob_remain[a]
                transitions[self.num_s-1, self.num_s-1, 0, a] = 1
            elif transition_type == 3:
                for s in range(self.num_s - 2):
                    transitions[s, s, 0, a] = prob_remain[a]
                    transitions[s, s+1, 0, a] = 2 * (1 - prob_remain[a]) / 3
                    transitions[s, s+2, 0, a] = (1 - prob_remain[a]) / 3
                transitions[self.num_s-2, self.num_s-2, 0, a] = prob_remain[a]
                transitions[self.num_s-2, self.num_s-1, 0, a] = 1 - prob_remain[a]
                transitions[self.num_s-1, self.num_s-1, 0, a] = 1
            elif transition_type == 4:
                for s in range(self.num_s-1):
                    transitions[s, s, 0, a] = prob_remain[a]
                    transitions[s, s+1:self.num_s, 0, a] = ((1 - prob_remain[a]) / (self.num_s-s)) \
                                                        * np.ones([self.num_s-s])
                transitions[self.num_s-1, self.num_s-1, 0, a] = 1

        return transitions


def get_state_list(num_states, num_arms):
    """ A helper function used to get state list: cartesian s^D.

    Example (3 states, 2 groups):
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    Returns:
        state_list: state tuple list
    """
    # generate state indices
    state_indices = np.arange(num_states)
    # get cartesian product
    state_indices_cartesian = np.meshgrid(*([state_indices] * num_arms), indexing="ij")
    # reshape and convert to list
    state_list = (np.stack(state_indices_cartesian, axis=-1).reshape(-1, num_arms).tolist())

    return state_list


# def state_decoding(num_states, num_arms, state_index):
#     sys_states = np.zeros(num_arms, dtype=int)
#     sys_states[num_arms-1] = np.mod(state_index+1, num_states)
#     sys_states[0] = (state_index-sys_states[num_arms-1]) // (num_states**(num_arms-1)) + 1
#     for ar in range(1, num_arms-1):
#         val = sys_states[num_arms-1]
#         for x in range(ar-1):
#             val = val + (sys_states[x]-1)*(num_states**(num_arms-x+1))
#         sys_states[ar] = (state_index-val) // (num_states**(num_arms-ar+1)) + 1
#     return sys_states

