import chainer.functions as F
import chainer.cuda as cuda
import chainer.optimizers as optimizers
import matplotlib as mpl
from chainer import serializers
from chainer import Variable
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
    avgs = [0]
    for n in range(len(rewards)):
        avg = (avgs[-1]*n + rewards[n]) / (n+1)
        avgs.append(avg)

    plt.figure()
    plt.plot(rewards, label="score")
    plt.plot(avgs[1:], label="avg")
    plt.legend()
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


def process_data(data):
    cumul_loss = 0
    for cur_state, next_state, old_q, taken_action, reward in data:
        new_q = old_q.data.copy()

        _, _, max_next_Q = compute_action(next_state, deterministic=True)

        new_q[0][taken_action] = reward + args.gamma * max_next_Q

        loss_prog = F.mean_squared_error(old_q, Variable(new_q))

        prog_net.cleargrads()
        loss_prog.backward()
        prog_optimizer.update()
        cumul_loss += loss_prog.data

    tqdm.write("Trained on {} samples \t Loss: {}".format(len(data), cumul_loss/len(data)))
    return cumul_loss / len(data)


def discount_reward(memory, cur_reward):
    discounted_reward = np.zeros(len(memory))
    discounted_reward[-1] = cur_reward[-1]
    for t in reversed(range(0, len(memory) - 1)):
        discounted_reward[t] = args.decay_rate * discounted_reward[t+1]

    return discounted_reward


def preprocess_obs(obs):
    # TODO: Maybe trim the unessential parts out of the obs
    # TODO: Turn obs to grayscale?
    obs = np.array(obs)  # Convert to potential cupy array
    obs = obs.reshape((1, 210, 160, 3))
    # obs = obs[:, :, :, 0]  # Remove the non-red colors for dimensionality reduction?

    return obs


def train():
    tqdm.write("observation space:", env.observation_space)
    tqdm.write("action space:", env.action_space)

    rewards = []
    loss = []

    for game_nr in tqdm(range(n_epoch)):
        prev_obs = None
        cur_obs = env.reset()

        batch = []
        memory = []
        running_reward = 0

        while True:
            if not headless:
                env.render()

            cur_obs = preprocess_obs(cur_obs)
            obs = cur_obs - prev_obs if prev_obs is not None else np.zeros(cur_obs.shape)
            prev_obs = cur_obs

            q_values, do_action, _ = compute_action(obs, deterministic=False)

            cur_obs, reward, done, info = env.step(do_action)

            if reward == 0:  # No reward in previous state
                memory.append([prev_obs, cur_obs, q_values, do_action])
            else:
                processed_memory = np.hstack((memory, discount_reward(memory, reward)))
                memory.clear()
                np.vstack((batch, processed_memory))  # append to training data

            running_reward += reward

            if done:
                tqdm.write("Game {}: {}".format(str(game_nr), str(running_reward)))
                rewards.append(running_reward)
                if game_nr % args.update_threshold == 0:
                    loss.append(process_data(batch))
                    batch.clear()
                break

    serializers.save_hdf5(args.outfile, prog_net)
    summary(rewards, loss)


def compute_action(obs, deterministic):
    """
    Computes the next action in an e-greedy fashion
    :param obs:
    :param deterministic:
    :return: 
    """
    q_values = prog_net(obs, 1)

    if np.random.rand() < epsilon and not deterministic:
        do_action = env.action_space.sample()
    else:
        if cuda.available:
            do_action = 1
            max_Q = -99999999
            for index, Q in enumerate(q_values.data[0]):
                if Q > max_Q:
                    do_action = index
                    max_Q = Q
        else:
            do_action = np.argmax(q_values.data[0])

    return q_values, do_action, q_values.data[0][do_action]


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
    parser.add_argument("--eta", dest="eta", type=int, default=0.001,
                        help="Learning Rate")
    parser.add_argument("--gamma", dest="gamma", type=int, default=0.99,
                        help="Discount factor")
    parser.add_argument("--decay-rate", dest="decay_rate", type=int, default=0.99,
                        help="Decay rate for future rewards")
    parser.add_argument("--update-after", dest="update_threshold", type=int, default=100,
                        help="Number of games needed to update")
    parser.add_argument("--headless", type=str2bool, nargs='?', const=True, default=True,
                        help="Headless mode, suppresses rendering and plotting")
    args = parser.parse_args()

    if cuda.available:  # GPU optimization
        import cupy as np
        np.cuda.Device(0).use()
        mpl.use('Agg')  # Suppress rendering

    import matplotlib.pyplot as plt
    env = gym.make(args.env)
    headless = args.headless
    # env = gym.make("SpaceInvaders-v4")

    number_of_actions = env.action_space.n
    epsilon = 0.1
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
