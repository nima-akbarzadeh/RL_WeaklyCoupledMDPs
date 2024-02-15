import numpy as np
from params import CostReward, MarkovDynamics


class WeaklyCoupled:
    def __init__(
            self,
            num_arms: int,
            num_states: int,
            reset_coef: float,
            function_type: str,
            prob_remain,
            transition_type=1,
            num_steps=1000,
    ):

        # Parameters
        self.num_arms = num_arms
        self.num_states = num_states
        self.num_steps = num_steps
        self.num_actions = self.num_arms + 1

        # Basic Functions
        costreward_class = CostReward(num_arms, num_states, function_type, reset_coef)
        self.reward_set = costreward_class.rewards
        dyn_class = MarkovDynamics(num_arms, num_states, prob_remain, transition_type)
        self.transitions = dyn_class.transitions

        # Initialization
        self.n_runs = 0
        self.step_counter = 0
        self.reward_info = []
        self.observation = self.reset()

    def reset(self):
        self.n_runs += 1
        self.step_counter = 0
        self.reward_info = []
        return np.zeros(self.num_arms, dtype=int)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (`list`): the action taken for each arm in the state list.

        Returns:
            state (`np.array`): the state of the environment after taking the action.
            reward (`float`): the reward for taking the action.
            done (`bool`): whether the episode is done.
            truncated ('bool): whether the episode is truncated.
            info (`dict`): additional information about the environment.

        """
        # Next state & Rewards & Info
        observation_ = np.copy(self.observation)
        reward = np.zeros(self.num_arms)
        info = []
        for a in range(self.num_arms):
            transition_prob = self.transitions[self.observation[a], :, action[a], a]
            observation_[a] = np.random.choice(np.arange(len(transition_prob)),
                                                  p=transition_prob)
            reward[a] = self.reward_set[self.observation[a], action[a], a]
            info.append({f"reward_{a}": reward[a]})
        self.observation = observation_

        # Done
        done = self.step_counter >= self.num_steps

        # Counter
        self.step_counter += 1

        return self.observation, sum(reward), done, False, info
