This projects a new deep Q-learning Network (DQN) structure to learn the optimal policy in weakly coupled Markov decision processes (MDP).

The neural network for the critic network is a sparse architecture where we have feed-forward layers for each MDP which generates an index given the current state.

The indices are then passes through another feed-forward layer to generate the Q-values. The best action is the one which results in the highest Q-value.

The structure tries to mimic the way Whittle indices are being generated and arms are being selected.
