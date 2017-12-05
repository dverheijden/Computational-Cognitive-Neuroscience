import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gym
import networks
from model import Model


def train():
	env = gym.make("Pong-v0")
	print("observation space:", env.observation_space)
	print("action space:", env.action_space)

	obs = env.reset()
	env.render()
	print('initial observation:', obs)

	action = env.action_space.sample()
	obs, r, done, info = env.step(action)

	print("next observation:,", obs)
	print("reward:", r)
	print("done:", done)
	print("info:", info)


if __name__ == "__main__":
	n_iter = 20

	prog_net = networks.ProgNet(n_actions=2)

	prog_model = Model(prog_net, lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)

	prog_optimizer = optimizers.SGD()
	prog_optimizer.setup(prog_model)

	print("Model Set Up!")

	train()
