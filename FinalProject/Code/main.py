import chainer.functions as F
import chainer.cuda as cuda
import chainer.optimizers as optimizers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from chainer import serializers
import numpy
import numpy as np
from tqdm import tqdm
import gym
import atari_py
import networks
from model import Model
import time


def summary(rewards):
    plt.plot(rewards)
    plt.xlabel("Game")
    plt.ylabel("Reward")
    plt.title("Reward as a function of nr. of games")
    plt.savefig("result/summary_{}.png".format(time.strftime("%d-%m-%Y %H:%M:%S")), format="png")

    if not headless:
        plt.show()


def train():
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    i = 0
    eta = 0.1

    rewards = []

    cumul_reward = 0
    n_epoch = 1200
    for i in tqdm(range(n_epoch)):
        obs = np.array(env.reset())
        while True:
            if not headless:
                env.render()
            # print('initial observation:', obs)
            
            obs = obs.reshape((1,3,210,160))
            # obs = obs.reshape((1,3,200,200))

            action, do_action = compute_action(obs)

            obs, reward, done, info = env.step(do_action)
            obs = np.array(obs)
            q_value = action
            new_q = q_value.data
            new_q[0][do_action] *= eta
            new_q[0][do_action] += reward

            # print(q_value)
            # print(new_q)
            loss = F.mean_squared_error(q_value, new_q)

            prog_model.predictor.cleargrads()
            loss.backward()
            prog_optimizer.update()

            cumul_reward += reward
            if done:
                tqdm.write(str(cumul_reward))
                rewards.append(cumul_reward)
                cumul_reward = 0
                break

        # print("next observation:,", obs)
        # print("reward:", r)
        # print("done:", done)
        # print("info:", info)
    serializers.save_hdf5('my_model.model', prog_model)
    summary(rewards)


def compute_action(obs):
    action = prog_model.predict(obs, 1)

    if cuda.available:
        do_action = 1
        max_Q = -99999999
        for index, Q in enumerate(action.data[0]):
            if Q > max_Q:
                do_action = index
    else:
        do_action = np.argmax(action.data[0])

    if epsilon > np.random.rand():
        do_action = env.action_space.sample()

    return action, do_action


def run_saved(filename):
    return serializers.load_hdf5(filename, networks.ProgNet)


if __name__ == "__main__":
    if cuda.available:
        import cupy as np
        np.cuda.Device(0).use()
    env = gym.make("Pong-v0")
    headless = True
    # env = gym.make("SpaceInvaders-v4")

    number_of_actions = env.action_space.n
    n_iter = 20
    epsilon = 0.2

    prog_model = Model(networks.ProgNet(n_actions=number_of_actions), lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)
    # prog_model = run_saved('my_model.model')

    if cuda.available:
        prog_model.to_gpu(0)

    prog_optimizer = optimizers.SGD()
    prog_optimizer.setup(prog_model)

    print("Model Set Up!")

    train()
