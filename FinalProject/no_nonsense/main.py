import chainer.functions as F
import chainer.optimizers as optimizers
import chainer.optimizer as optimizer
import matplotlib as mpl
from chainer import serializers
from chainer import Variable
import chainer
import numpy as np
from gym.envs import frameskip
from tqdm import tqdm
import gym
from networks import FCN, CNN
import time
import argparse
from random import shuffle, randint
from decimal import Decimal
import math
from copy import deepcopy
import chainer.computational_graph as c
import os
from wrappers import FrameStackWrapper, ResetLifeLostWrapper


def summary(prefix):
    rewards = np.genfromtxt(prefix + "rewards.csv")
    loss = np.genfromtxt(prefix + "loss.csv")

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
    plt.savefig("results/{}/{}_summary_reward.png".format(args.env, time.strftime("%Y-%m-%d-%H:%M")), format="png")
    plt.savefig("results/{}/{}_summary_reward.eps".format(args.env, time.strftime("%Y-%m-%d-%H:%M")), format="eps")

    if not args.headless:
        plt.show()

    plt.figure()
    plt.plot(loss)
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("Loss as a function of nr. of updates")
    plt.savefig("results/{}/{}_summary_loss.png".format(args.env, time.strftime("%Y-%m-%d-%H:%M")), format="png")
    plt.savefig("results/{}/{}_summary_loss.eps".format(args.env, time.strftime("%Y-%m-%d-%H:%M")), format="eps")

    if not args.headless:
        plt.show()

    serializers.save_hdf5("results/{}/{}".format(args.env, args.outfile), net)


def process_data(data):
    """
    Process data by computing their new Q (target) values, compute the (Huber) loss and backprop
    We circumvent the use of a target network by pre-computing the targets
    :param data:
    :return:
    """

    data = np.array(data)
    cur_states = np.stack(data[:, 0], axis=1)
    cur_states = np.squeeze(cur_states, axis=0)
    cur_states = chainer.cuda.to_gpu(cur_states) if chainer.cuda.available else cur_states
    cur_qs = net(cur_states)

    next_states = np.stack(data[:, 1], axis=1)
    next_states = np.squeeze(next_states, axis=0)
    next_states = chainer.cuda.to_gpu(next_states) if chainer.cuda.available else next_states
    next_qs = net(next_states)

    with chainer.no_backprop_mode():  # Don't think this actually does anything, but better safe than sorry
        target_qs = deepcopy(cur_qs.data)
        target_qs = chainer.cuda.to_cpu(target_qs) if chainer.cuda.available else target_qs
        # Memory Efficiency
        # taken_action = data[:, 2]
        # terminal = data[:, 3]
        # rewards = data[:, 4]
        for i in range(len(target_qs)):
            # Q(s, a) = r + gamma * max_a(Q(s',a')) if game not over else r
            target_qs[i, data[i, 2]] = data[i, 4] + args.gamma * next_qs.data[i, :].max() if not data[i, 3] else data[i, 2]

    target_qs = chainer.cuda.to_gpu(target_qs) if chainer.cuda.available else target_qs
    loss = F.squared_error(cur_qs, target_qs)

    # Debugging Information
    # g = c.build_computational_graph(loss)
    # with open('computational_graph.graph', 'w') as o:
    #     o.write(g.dump())

    loss = F.mean(loss)
    net.cleargrads()
    loss.backward()
    optim.update()

    return float(loss.data)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) / 255


def preprocess_obs(obs, dim=3):
    """
    Process one observation to a 1D array
    :param obs:
    :return:
    """
    if args.env == "CartPole-v0":
        obs = obs.astype(np.float32)
        return np.expand_dims(obs, axis=0)

    processed_obs = []
    for frame in obs:
        frame = rgb2gray(frame)
        frame = frame.astype(np.float32)
        if args.env == "Pong-v0":
            frame = frame[35:195]  # crop
            frame = frame[::2, ::2]  # downsample by factor of 2
            # obs = obs[:, :, 0]  # use only only RED
            # obs[obs == 144] = 0  # erase background (background type 1)
            # obs[obs == 109] = 0  # erase background (background type 2)
            # obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1

        elif args.env == "Breakout-v0":
            frame = frame[35:195, 8:152]  # crop
            frame = frame[::2, ::2]  # downsample by factor of 2

        elif args.env == "SpaceInvaders-v0":
            frame = frame[:195, :]  # crop

        if dim == 2:  # For MLP input
            frame = frame.ravel()

        # elif dim == 3:  # For Conv input
        #     frame = np.expand_dims(frame, axis=0)

        processed_obs.append(frame)

    processed_obs = np.array(processed_obs)
    processed_obs = np.expand_dims(processed_obs, axis=0)

    return processed_obs


def discount_reward(memory, cur_reward):
    """
    This was used in traditional Q-learning, however in DQN this is not used. DQN relies on pure value iteration.
    The idea was that we discount rewards over states that had no (direct) reward. If a reward was gained, iterate over
    the previous states without reward and give them a discounted reward value.
    :param memory: game states
    :param cur_reward: gained reward
    :return:
    """
    discounted_reward = np.zeros(len(memory))
    discounted_reward[-1] = cur_reward
    for t in reversed(range(0, len(memory) - 1)):
        discounted_reward[t] = args.decay_rate * discounted_reward[t+1]

    return discounted_reward


def compute_action(obs, deterministic):
    """
    Computes the next action in an e-greedy fashion, where e is decaying over the total number of steps taken
    :param obs:
    :param deterministic: use epsilon or not
    :return: q_values of the observation, index of the best action, maximum of the computed q_values
    """
    obs = chainer.cuda.to_gpu(obs) if chainer.cuda.available else obs  # Copy to GPU
    q_values = net(obs)

    global eps_threshold
    eps_threshold = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-1 * total_moves / epsilon_decay)

    if np.random.rand() < eps_threshold and not deterministic:
        best_action = env.action_space.sample()
    else:
        best_action = F.argmax(q_values)
        best_action = best_action.data
        best_action = chainer.cuda.to_cpu(best_action) if chainer.cuda.available else best_action

    return q_values, best_action, F.max(q_values)


def train():
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    global total_moves

    prefix = "results/{}/{}_".format(args.env, time.strftime("%Y-%m-%d-%H:%M"))
    rewards_file = open(prefix + "rewards.csv", 'w')
    losses_file = open(prefix + "loss.csv", 'w')

    tqdm.write(" {:^5} | {:^10} | {:^5} | {:^10} | {:^12} | {:^10}\n".format(
        "Game", "Epsilon", "Score", "Score@100", "Total Moves", "Loss")
               + "-"*70)
    with open("{}.progress".format(args.env), 'w') as f:

        replay_memory = []
        loss_list = []
        rewards_list = []
        running_reward = 0  # games may span multiple episodes

        obs_dimension = 3 if not args.toy else 2

        for game_nr in tqdm(range(1, args.n_epoch+1), unit="game", ascii=True, file=f):
            cur_obs = preprocess_obs(env.reset(), dim=obs_dimension)

            moves = 0

            while True:
                moves += 1  # Bookkeeping
                total_moves += 1 if eps_threshold > epsilon_min else 0  # epsilon_decay

                if not args.headless:
                    env.render()

                _ , taken_action, _ = compute_action(cur_obs, deterministic=False)

                next_obs, reward, done, info = env.step(taken_action)
                next_obs = preprocess_obs(next_obs, dim=obs_dimension)
                if reward != 0:
                    reward = -1 if reward < 0 else 1

                running_reward += reward

                # Simple DQN approach - Pin it on Value Iteration / Reward Propagation
                if len(replay_memory) is args.replay_size:
                    replay_memory[randint(0, len(replay_memory) - 1)] = [cur_obs, next_obs, taken_action, done,
                                                                         reward]
                else:
                    replay_memory.append([cur_obs, next_obs, taken_action, done, reward])

                # Update after every args.update_threshold frames
                if moves % args.update_threshold is 0 and len(replay_memory) > args.replay_size_min:
                    train_data = [replay_memory[randint(0, len(replay_memory) - 1)] for _ in range(args.batch_size)]
                    loss = process_data(train_data)
                    loss_list.append(loss)
                    losses_file.write("{}\n".format(loss))

                if info['done']:  # Running out of lives means the game is done
                    rewards_file.write("{}\n".format(running_reward))
                    if len(rewards_list) is 100:
                        del rewards_list[0]  # Only store last 100 rewards
                    rewards_list.append(running_reward)

                    if game_nr % args.plot_every is 0:
                        game_loss = float('NaN')
                        avg_reward = sum(rewards_list)/len(rewards_list)

                        if len(loss_list) > 0:
                            game_loss = sum(loss_list) / len(loss_list)
                            loss_list.clear()

                        tqdm.write(" {:5d} | {:10.8f} | {:+5.0f} | {:10.5f} | {:12d} | {:^10f}".format(
                            game_nr, eps_threshold, running_reward, avg_reward, moves, game_loss
                            )
                        )
                    running_reward = 0
                    break

    summary(prefix)


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
    parser.add_argument("--epochs", dest="n_epoch", type=int, default=2000,
                        help="Amount of epochs")
    parser.add_argument("--replay-size", dest="replay_size", type=int, default=1000000,
                        help="Size of replay buffer")
    parser.add_argument("--replay-size-initial", dest="replay_size_min", type=int, default=2000,
                        help="Size of replay buffer")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32,
                        help="Size of one training batch")
    parser.add_argument("--frames", dest="frames", type=int, default=4,
                        help="Number of stacked frames per observation")
    parser.add_argument("--alpha", dest="alpha", type=float, default=2.5e-4,
                        help="Learning Rate")
    parser.add_argument("--momentum", dest="momentum", type=float, default=0.95,
                        help="Momentum")
    parser.add_argument("--epsilon-min", dest="epsilon_min", type=float, default=0.05,
                        help="Minimum probability of doing a random action")
    parser.add_argument("--epsilon-max", dest="epsilon_max", type=float, default=1,
                        help="Maximum probability of doing a random action")
    parser.add_argument("--epsilon-decay", dest="epsilon_decay", type=float, default=50000,
                        help="Measure at which epsilon decays (very game dependent!)")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--decay-rate", dest="decay_rate", type=float, default=0.99,
                        help="Decay rate for future rewards")
    parser.add_argument("--update-after", dest="update_threshold", type=int, default=4,
                        help="Number of frames needed to update")
    parser.add_argument("--plot-every", dest="plot_every", type=int, default=1,
                        help="Number of games before showing summary")
    parser.add_argument("--headless", action="store_true", required=False,
                        help="Headless mode, suppresses rendering and plotting")
    parser.add_argument("--toy", action='store_true', required=False,
                        help="Create shallow networks")
    args = parser.parse_args()
    print(args)

    if args.headless:  # GPU optimization
        mpl.use('Agg')  # Suppress rendering

    import matplotlib.pyplot as plt

    env = gym.make(args.env)
    if not args.toy:
        env = FrameStackWrapper(env, args.frames)
        env = ResetLifeLostWrapper(env)

    epsilon_max = args.epsilon_max
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay
    eps_threshold = 1

    total_moves = 0

    net = CNN(conv_channels=args.frames, n_actions=env.action_space.n) if not args.toy \
        else FCN(n_actions=env.action_space.n)
    if chainer.cuda.available:
        net.to_gpu()
    optim = optimizers.RMSpropGraves(lr=args.alpha, momentum=args.momentum)
    optim.setup(net)

    if not os.path.exists("results/{}".format(args.env)):
        os.makedirs("results/{}".format(args.env))

    train()




