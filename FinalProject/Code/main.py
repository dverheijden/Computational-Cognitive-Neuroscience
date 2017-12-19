import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib.pyplot as plt
from chainer import serializers
import numpy as np
from tqdm import tqdm
import gym
import atari_py
import networks
from model import Model


def train():
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    obs = env.reset()
    
    i = 0
    eta = 0.1
    cumul_reward = 0
    n_epoch = 100
    for i in tqdm(range(n_epoch)):
        while True:
            # env.render()
            # print('initial observation:', obs)
            
            obs = obs.reshape((1,3,210,160))
            # obs = obs.reshape((1,3,200,200))

            action, do_action = compute_action(obs)
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

            cumul_reward += reward
            if done:
                tqdm.write(str(cumul_reward))
                cumul_reward = 0
                env.reset()
                break

        serializers.save_hdf5('my_net', prog_net)
        # print("next observation:,", obs)
        # print("reward:", r)
        # print("done:", done)
        # print("info:", info)

def compute_action(obs):
    action = prog_net(obs, 1)
    do_action = np.argmax(action.data[0])

    if epsilon > np.random.rand():
        do_action = env.action_space.sample()


    return action, do_action


def run_saved(filename):
    return serializers.load_hdf5(filename, networks.ProgNet)


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    # env = gym.make("SpaceInvaders-v4")

    number_of_actions = env.action_space.n
    n_iter = 20
    epsilon = 0.2

    prog_net = networks.ProgNet(n_actions=number_of_actions)
    # prog_net = run_saved('my_net')
    
    prog_model = Model(prog_net, lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)

    prog_optimizer = optimizers.SGD()
    prog_optimizer.setup(prog_model)

    print("Model Set Up!")

    train()
