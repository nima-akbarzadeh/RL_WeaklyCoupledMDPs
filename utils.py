import numpy as np
import matplotlib.pyplot as plt


def plot_learning_scores(x, scores, filename, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(x, running_avg)
    plt.savefig(filename)
