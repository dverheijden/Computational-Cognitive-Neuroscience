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
from wrappes import FrameStackWrapper, ResetLifeLostWrapper


def summary(rewards, loss):
    if not os.path.exists("results/{}".format(args.env)):
        os.makedirs("results/{}".format(args.env))

    with open("results/{}/{}_reward.csv".format(args.env, time.strftime("%Y-%m-%d-%H:%M")), 'w') as r_file:
        for r in rewards:
            r_file.write("{}\n".format(r))

    with open("results/{}/{}_loss.csv".format(args.env, time.strftime("%Y-%m-%d-%H:%M")), 'w') as l_file:
        for l in loss:
            l_file.write("{}\n".format(l))

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
    cur_qs.name = "current Q-Values"

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

    obs = rgb2gray(obs)
    obs = obs.astype(np.float32)
    if args.env == "Pong-v0":
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2]  # downsample by factor of 2
        # obs = obs[:, :, 0]  # use only only RED
        # obs[obs == 144] = 0  # erase background (background type 1)
        # obs[obs == 109] = 0  # erase background (background type 2)
        # obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1

    elif args.env == "Breakout-v0":
        obs = obs[35:195, 8:152]  # crop
        obs = obs[::2, ::2]  # downsample by factor of 2

    elif args.env == "SpaceInvaders-v0":
        obs = obs[:195, :]  # crop

    if dim == 2:  # For MLP input
        obs = obs.ravel()
    elif dim == 3:  # For Conv input
        obs = np.expand_dims(obs, axis=0)

    obs = np.expand_dims(obs, axis=0)
    return obs


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

    rewards = []
    losses = []
    tqdm.write(" {:^5} | {:^10} | {:^5} | {:^10} | {:^10} | {:^12} | {:^10} | {:^15}\n".format(
        "Game", "Epsilon", "Score", "Total Avg", "Score@10", "Total Moves", "Loss", "Avg Loss")
               + "-"*100)
    with open("{}.progress".format(args.env), 'w') as f:

        replay_memory = []
        loss_list = []

        obs_dimension = 3 if not args.toy else 2

        for game_nr in tqdm(range(1, args.n_epoch+1), unit="game", ascii=True, file=f):
            prev_obs = None
            cur_obs = preprocess_obs(env.reset(), dim=obs_dimension)

            running_reward = 0

            moves = 0

            while True:
                moves += 1  # Bookkeeping
                total_moves += 1  # epsilon_decay

                if not args.headless:
                    env.render()

                if not args.env == "CartPole-v0":
                    state = np.subtract(cur_obs, prev_obs) if prev_obs is not None else np.zeros(cur_obs.shape, dtype=np.float32)
                else:
                    state = cur_obs

                prev_obs = cur_obs

                _ , taken_action, _ = compute_action(state, deterministic=False)

                cur_obs, reward, done, info = env.step(taken_action)
                cur_obs = preprocess_obs(cur_obs, dim=obs_dimension)

                if reward != 0:
                    reward = -1 if reward < 0 else 1

                if args.env != "CartPole-v0":
                    state_future = np.subtract(cur_obs, prev_obs)
                else:
                    state_future = cur_obs

                running_reward += reward

                # Simple DQN approach - Pin it on Value Iteration / Reward Propagation
                if len(replay_memory) is args.replay_size:
                    replay_memory[randint(0, len(replay_memory)-1)] = [state, state_future, taken_action, done, reward]
                else:
                    replay_memory.append([state, state_future, taken_action, done, reward])

                # Update after every args.update_threshold frames
                if moves % args.update_threshold is 0 and len(replay_memory) > args.replay_size_min:
                    train_data = [replay_memory[randint(0, len(replay_memory) - 1)] for _ in range(args.batch_size)]
                    loss = process_data(train_data)
                    loss_list.append(loss)
                    losses.append(loss)

                if done:
                    rewards.append(running_reward)
                    game_loss = float('NaN')
                    avg_loss = float('NaN')

                    if game_nr % args.plot_every is 0:
                        # Update after X frames Method
                        if len(loss_list) > 0:
                            game_loss = sum(loss_list) / len(loss_list)
                            avg_loss = sum(losses) / len(losses)
                            loss_list.clear()

                        tqdm.write(" {:5d} | {:10.8f} | {:+5.0f} | {:10.5f} | {:10.5f} | {:12d} | {:^10f} | {:^15f}".format(
                            game_nr, eps_threshold, running_reward, sum(rewards)/len(rewards),
                            sum(rewards[-10:])/len(rewards[-10:]),
                            moves, game_loss, avg_loss
                            )
                        )
                    break

    summary(rewards, losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Neural Network")
    parser.add_argument("--model-name", dest="model_path",
                        help="Path to a pretrained model")
    parser.add_argument("--output", dest="outfile",
                        help="Path to output model")
    parser.add_argument("--env", dest="env",
                        help="Environment Name", default="SpaceInvaders-v0")
    parser.add_argument("--hidden", dest="n_hidden", type=int, default=256,
                        help="Amount of hidden units")
    parser.add_argument("--feature-maps", dest="n_feature_maps", type=int, default=12,
                        help="Amount of feature maps")
    parser.add_argument("--epochs", dest="n_epoch", type=int, default=2000,
                        help="Amount of epochs")
    parser.add_argument("--replay-size", dest="replay_size", type=int, default=100000,
                        help="Size of replay buffer")
    parser.add_argument("--replay-size-initial", dest="replay_size_min", type=int, default=2000,
                        help="Size of replay buffer")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128,
                        help="Size of Batch Size")
    parser.add_argument("--frames", dest="frames", type=int, default=4,
                        help="Number of stacked frames per observation")
    parser.add_argument("--alpha", dest="alpha", type=float, default=1e-5,
                        help="Learning Rate")
    parser.add_argument("--epsilon-min", dest="epsilon_min", type=float, default=0.05,
                        help="Minimum probability of doing a random action")
    parser.add_argument("--epsilon-max", dest="epsilon_max", type=float, default=0.8,
                        help="Maximum probability of doing a random action")
    parser.add_argument("--epsilon-decay", dest="epsilon_decay", type=float, default=24000,
                        help="Measure at which epsilon decays (very game dependent!)")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--decay-rate", dest="decay_rate", type=float, default=0.99,
                        help="Decay rate for future rewards")
    parser.add_argument("--update-after", dest="update_threshold", type=int, default=50,
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

    net = CNN(n_actions=env.action_space.n) if not args.toy \
        else FCN(conv_channels=args.frames, n_actions=env.action_space.n)
    if chainer.cuda.available:
        net.to_gpu()
    optim = optimizers.RMSprop(lr=args.alpha)
    optim.setup(net)

    train()




