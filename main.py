import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from env_wcmdp import WeaklyCoupled
from SparseDQN import Agent as SDQN_Agent
from DQN import Agent as DQN_Agent
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    num_arms = 3
    num_states = 3
    env = WeaklyCoupled(num_arms=num_arms, num_states=num_states, reset_coef=1.5,
                        function_type='linear', prob_remain=0.5*np.ones(num_arms),
                        transition_type=1, num_steps=500)
    gamma = 0.99

    epsilon = 1.0
    eps_min = 0.01
    eps_dec = 5e-4
    n_episodes = 20
    load_agent = False

    n_actions = num_arms+1
    agent1 = DQN_Agent(env, [num_arms], n_actions, gamma, epsilon, eps_min, eps_dec, n_episodes)
    agent2 = SDQN_Agent(env, num_arms, n_actions, num_states, gamma, epsilon, eps_min, eps_dec,
                        n_episodes)
    if load_agent:
        agent1.load_model()
        agent2.load_model()

    scores2 = agent2.train()
    scores1 = agent1.train()

    fig, ax = plt.subplots()
    ax.plot(range(len(scores1)), scores1, label="DQN")
    ax.plot(range(len(scores2)), scores2, label="SDQN")
    ax.set(xlabel='Episodes', ylabel='Discounted Reward', title='Learning Curve')
    ax.grid()
    plt.legend()
    plt.show()
    plt.savefig('dqn_wcmdp')
    plt.close()
