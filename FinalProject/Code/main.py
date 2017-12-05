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
	
	i = 0
	while True:
		i+=1
		env.render()
		# print('initial observation:', obs)

		obs = obs.reshape((1,3,210,160))
		action = np.argmax(prog_net(obs, 1).data[0])
		print(action)

		# action = env.action_space.sample()
		obs, r, done, info = env.step(action)
		if done:
			env.reset()
		# print("next observation:,", obs)
		# print("reward:", r)
		# print("done:", done)
		# print("info:", info)


if __name__ == "__main__":
	n_iter = 20

	
	prog_net = networks.ProgNet(n_actions=6)

	prog_model = Model(prog_net, lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)

	prog_optimizer = optimizers.SGD()
	prog_optimizer.setup(prog_model)

	print("Model Set Up!")

	train()
