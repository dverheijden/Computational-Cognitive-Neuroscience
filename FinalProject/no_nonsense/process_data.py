import matplotlib.pyplot as plt
import numpy as np
import math

data = np.genfromtxt("2018-01-24-0025_reward.csv")

N = len(data)

smooth_scale = 3
data = data[::smooth_scale]

data_smoothed = []
for index, d in enumerate(data):
    data_smoothed.append(np.mean(data[index-smooth_scale:index+smooth_scale])) \
        if index > smooth_scale else data_smoothed.append(d)

p = np.poly1d(np.polyfit(np.linspace(0, len(data), len(data)), data_smoothed, 100))

data_mean = []
for index, d in enumerate(data_smoothed):
    data_mean.append((data_smoothed[index] * (index+1) + d)/(index+2))

xlabels = np.linspace(0, N, len(data_smoothed))

plt.plot(xlabels, data_smoothed, label="smoothed rewards")
# plt.plot(xlabels, data_mean, label="avg reward")
plt.plot(xlabels, p(np.linspace(0, len(data), len(data))), label="avg reward")
plt.xlabel("Games")
plt.ylabel("Reward")
plt.title("Env: {}".format("CartPole-v0"))
plt.legend(loc="lower right")
plt.show()
