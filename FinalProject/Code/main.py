import chainer.functions as F
import chainer.cuda as cuda
import chainer.optimizers as optimizers
import matplotlib as mpl
from chainer import serializers
import numpy
import numpy as np
from tqdm import tqdm
import gym
import atari_py
import networks
from model import Model
import time
import argparse


def summary(rewards, loss):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Game")
    plt.ylabel("Reward")
    plt.title("Reward as a function of nr. of games")
    plt.savefig("result/summary_reward_{}.png".format(time.strftime("%d-%m-%Y %H:%M:%S")), format="png")

    if not headless:
        plt.show()

    plt.figure()
    plt.plot(loss)
    plt.xlabel("Game")
    plt.ylabel("Loss")
    plt.title("Loss as a function of nr. of games")
    plt.savefig("result/summary_loss_{}.png".format(time.strftime("%d-%m-%Y %H:%M:%S")), format="png")

    if not headless:
        plt.show()


def train():
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    rewards = []
    loss = []

    for _ in tqdm(range(n_epoch)):
        obs = np.array(env.reset())
        cumul_reward = 0
        cumul_loss = 0
        while True:
            if not headless:
                env.render()
            # print('initial observation:', obs)
            
            obs = obs.reshape((1, 3, 210, 160))
            # obs = obs.reshape((1,3,200,200))

            q_values, do_action = compute_action(obs)

            obs, reward, done, info = env.step(do_action)
            obs = np.array(obs)

            new_q = np.array(q_values.data).copy()
            new_q[do_action] *= eta
            new_q[do_action] += reward

            # print(q_value)
            # print(new_q)
            loss_prog = F.mean_squared_error(q_values, new_q)

            prog_net.cleargrads()
            loss_prog.backward()
            prog_optimizer.update()

            cumul_reward += reward
            cumul_loss += loss_prog.data
            if done:
                tqdm.write("Reward: {} \t Loss: {}".format(str(cumul_reward), str(cumul_loss)))
                rewards.append(cumul_reward)
                loss.append(cumul_loss)
                break

        # print("next observation:,", obs)
        # print("reward:", r)
        # print("done:", done)
        # print("info:", info)
    serializers.save_hdf5(args.outfile, prog_net)
    summary(rewards, loss)


def compute_action(obs):
    action = prog_net.predict(obs, 1)

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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Neural Network")
    parser.add_argument("--model-name", dest="model_path",
                        help="Path to a pretrained model")
    parser.add_argument("--output", dest="outfile",
                        help="Path to output model")
    parser.add_argument("--env", dest="env",
                        help="Environment Name", default="Pong-v0")
    parser.add_argument("--hidden", dest="n_hidden", type=int, default=256,
                        help="Amount of hidden units")
    parser.add_argument("--feature-maps", dest="n_feature_maps", type=int, default=12,
                        help="Amount of feature maps")
    parser.add_argument("--epochs", dest="n_epoch", type=int, default=20,
                        help="Amount of epochs")
    parser.add_argument("--eta", dest="eta", type=int, default=0.1,
                        help="Learning Rate")
    parser.add_argument("--headless", type=str2bool, nargs='?', const=True, default=True,
                        help="Headless mode, supresses rendering and plotting")
    args = parser.parse_args()

    if cuda.available: # Server optimization
        import cupy as np
        np.cuda.Device(0).use()
        mpl.use('Agg')

    import matplotlib.pyplot as plt
    env = gym.make(args.env)
    headless = args.headless
    # env = gym.make("SpaceInvaders-v4")

    number_of_actions = env.action_space.n
    epsilon = 0.2
    eta = args.eta

    n_epoch = args.n_epoch
    if args.model_path:
        prog_net = run_saved(args.model_path)
    else:
        prog_net = networks.ProgNet(n_actions=number_of_actions, n_feature_maps=args.n_feature_maps,
                                    n_hidden_units=args.n_hidden)

    if cuda.available:
        prog_net.to_gpu(0)

    prog_optimizer = optimizers.SGD()
    prog_optimizer.setup(prog_net)

    print("Model Set Up!")

    train()
