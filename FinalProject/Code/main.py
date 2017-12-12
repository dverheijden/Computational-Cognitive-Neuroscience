import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gym
import atari_py
import networks
from model import Model


def train():
	env = gym.make("Pong-v0")
	print("observation space:", env.observation_space)
	print("action space:", env.action_space)

	obs = env.reset()
	
	i = 0
	eta = 0.1

	while True:
		i+=1
		env.render()
		# print('initial observation:', obs)

		obs = obs.reshape((1,3,210,160))
		# obs = obs.reshape((1,3,200,200))

		action = prog_net(obs, 1)
		
		do_action = np.argmax(prog_net(obs, 1).data[0])
		# print(action)

		# action = env.action_space.sample()
		obs, reward, done, info = env.step(do_action)

		q_value = action
		new_q = q_value.data
		new_q[0][do_action] *= eta
		new_q[0][do_action] += reward

		# print(q_value)
		# print(new_q)
		loss = F.mean_squared_error(q_value, new_q)

		prog_net.cleargrads()
		loss.backward()
		prog_optimizer.update()

		print(reward)
		
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
