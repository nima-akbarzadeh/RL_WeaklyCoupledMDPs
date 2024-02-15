import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden1_dims, hidden2_dims, filename, chkpt_dir,
                 device=DEVICE):
        super(DeepQNetwork, self).__init__()

        # Initialize the network
        self.fc_net = nn.Sequential(
            nn.Linear(*input_dims, hidden1_dims),
            nn.ReLU(),
            nn.Linear(hidden1_dims, hidden2_dims),
            nn.ReLU(),
            nn.Linear(hidden2_dims, n_actions)
        )

        # Initialize the rest of parameters
        self.device = device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)

    def forward(self, state):
        return self.fc_net(state)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ReplayBuffer:
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.bool8)

    def store_data(self, state, action, reward, state_, terminal):
        index = self.mem_counter % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.reward_mem[index] = reward
        self.action_mem[index] = action
        self.terminal_mem[index] = terminal

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminals


class Agent:
    def __init__(self, env, input_dims, n_actions, gamma, epsilon, eps_min, eps_dec, n_episodes,
                 lr=1e-3, batch_size=64, hidden1_dims=256, hidden2_dims=256, mem_size=100000,
                 chkpt_dir='./tmp/dqn'):
        self.env = env
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.mem_size = mem_size

        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_net = DeepQNetwork(input_dims, n_actions, hidden1_dims, hidden2_dims,
                                  'wcdqn_restart', chkpt_dir)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Choose action according to the Q-network
            self.q_net.eval()
            state = torch.tensor(observation, dtype=torch.float32).to(self.q_net.device)
            actions = self.q_net.forward(state)
            self.q_net.train()
            return torch.argmax(actions).item()
        else:
            # Choose action randomly
            return np.random.choice(self.action_space)

    def store_data(self, state, action, reward, state_, terminal):
        self.memory.store_data(state, action, reward, state_, terminal)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        self.q_net.save_checkpoint()

    def load_model(self):
        self.q_net.load_checkpoint()

    def get_targets(self, rewards, states_, terminals, indices):
        # 1. Get the sampled Q-values for the next sampled states
        # 2. Set the next sampled Q-values to 0 if the state is terminal
        # 3. Choose the best action for those states
        q_preds_ = self.q_net.forward(states_)
        q_preds_[terminals] = 0.0
        actions_ = torch.argmax(q_preds_, dim=1)

        return rewards + self.gamma * q_preds_[indices, actions_]

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # Sample memory and convert it to tensors
        states, actions, rewards, new_states, terminals = self.memory.sample_buffer(self.batch_size)
        states = torch.tensor(states).to(self.q_net.device)
        actions = torch.tensor(actions).to(self.q_net.device)
        rewards = torch.tensor(rewards).to(self.q_net.device)
        states_ = torch.tensor(new_states).to(self.q_net.device)
        terminals = torch.tensor(terminals).to(self.q_net.device)

        # index the batch elements
        indices = np.arange(self.batch_size)

        # Compute the target values
        q_targets = self.get_targets(rewards, states_, terminals, indices)

        # Compute the current q-values
        q_preds = self.q_net.forward(states)[indices, actions]

        # Compute the loss
        loss = self.loss(q_preds, q_targets).to(self.q_net.device)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease the epsilon if possible
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def train(self):
        scores = []
        for i in range(self.n_episodes):
            score = 0
            terminal = False
            observation = self.env.reset()
            while not terminal:
                action = self.choose_action(observation)

                # One-hot encoding of the action
                action_list = np.zeros(self.env.num_arms, dtype=int)
                if action != 0:
                    action_list[action - 1] = 1

                observation_, reward, done, truncated, info = self.env.step(action_list)
                score += reward
                terminal = done or truncated
                self.store_data(observation, action, reward, observation_, terminal)
                self.learn()
                observation = observation_

            scores.append(score)
            avg_score = np.mean(scores[-100:])

            print(
                'episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
                               'epsilon %.2f' % self.epsilon
            )

        return scores
