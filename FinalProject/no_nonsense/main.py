import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib as mpl
import matplotlib.pyplot as plt
from chainer import serializers
from chainer import Variable
import chainer
import numpy as np
from tqdm import tqdm
import gym
from networks import FCN
import time
import argparse
from random import shuffle, randint
from decimal import Decimal
import math


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

    if not args.headless:
        plt.show()

    plt.figure()
    plt.plot(loss)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss as a function of nr. of games")
    plt.savefig("result/summary_loss_{}.png".format(time.strftime("%d-%m-%Y %H:%M:%S")), format="png")

    if not args.headless:
        plt.show()


def process_data(data):
    cumul_loss = 0
    for cur_state, next_state, old_q, taken_action, reward, terminal in data:

        with chainer.no_backprop_mode():
            new_q = old_q.data.copy()

            _, _, max_next_Q = compute_action(next_state, deterministic=True)

            new_q[0][taken_action] = reward + args.gamma * max_next_Q if not terminal else reward

        loss = F.huber_loss(old_q, new_q, 1)
        net.cleargrads()
        loss.backward()
        optim.update()
        cumul_loss += loss.data

    # tqdm.write("Trained on {} samples \t Loss: {}".format(len(data), cumul_loss))
    return float(cumul_loss / len(data)), float(cumul_loss)


def preprocess_obs(obs):
    """
    Process one observation to a 1D array
    :param obs:
    :return:
    """
    obs = obs[35:195]  # crop
    obs = obs[::2, ::2, 0]  # downsample by factor of 2 and take only RED
    obs[obs == 144] = 0  # erase background (background type 1)
    obs[obs == 109] = 0  # erase background (background type 2)
    obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1

    obs = obs.astype(np.float32).ravel()
    obs = np.expand_dims(obs, axis=0)

    return obs


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * args.dacay_rate + r[t]
        discounted_r[t] = running_add
    return discounted_r


def compute_action(obs, deterministic):
    """
    Computes the next action in an e-greedy fashion
    :param obs:
    :param deterministic:
    :return:
    """
    obs = chainer.cuda.to_gpu(obs) if chainer.cuda.available else obs
    q_values = net(obs)

    global eps_threshold
    eps_threshold = epsilon_min + (epsilon_max - epsilon_min) * math.exp(-1 * total_moves / epsilon_decay)

    if np.random.rand() < eps_threshold and not deterministic:
        best_action = env.action_space.sample()
    else:
        best_action = np.argmax(q_values.data[0])

    return q_values, best_action, q_values.data[0][best_action]


def train():
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    global total_moves

    rewards = []
    losses = []
    tqdm.write(" {:^5} | | {:^10} | {:^5} | {:^10} | {:^10} | {:^12} | {:^10} | {:^15}\n".format(
        "Game", "Epsilon", "Score", "Total Avg", "Score@10", "Total Moves", "Avg Loss", "Total Loss")
               + "-"*88)

    for game_nr in tqdm(range(1, args.n_epoch+1), unit="game", ascii=False):
        prev_obs = None
        cur_obs = preprocess_obs(env.reset())

        memory = []

        running_reward = 0

        moves = 0

        loss_list = []

        while True:
            moves += 1  # Bookkeeping
            total_moves += 1  # epsilon_decay

            if not args.headless:
                env.render()

            obs = np.subtract(cur_obs, prev_obs) if prev_obs is not None else np.zeros(cur_obs.shape, dtype=np.float32)
            prev_obs = cur_obs

            q_values, taken_action, _ = compute_action(obs, deterministic=False)

            cur_obs, reward, done, info = env.step(taken_action)
            cur_obs = preprocess_obs(cur_obs)
            running_reward += reward

            # Old Approach...
            # memory.append([prev_obs, cur_obs, q_values, taken_action])
            #
            # if reward != 0 or done:
            #     # Distribute discounted reward to previous actions
            #     discounted_reward = discount_reward(memory, reward)
            #     for i in range(len(memory)):
            #         memory[i].extend([discounted_reward[i]])
            #     batch.extend(memory)  # append to training data
            #     memory.clear()

            # Simple DQN approach
            if len(memory) is args.replay_size:
                memory[randint(0, len(memory)-1)] = [prev_obs, cur_obs, q_values, taken_action, reward, done]
            else:
                memory.append([prev_obs, cur_obs, q_values, taken_action, reward, done])

            # Update after every args.update_threshold frames
            if moves % args.update_threshold is 0 and len(memory) > args.batch_size:
                train_data = [memory[randint(0, len(memory) - 1)] for _ in range(args.batch_size)]
                avg_loss, _ = process_data(train_data)
                loss_list.append(avg_loss)

            if done:
                rewards.append(running_reward)
                # Old Approach
                # batch_loss = "N/A"
                # avg_loss = "N/A"
                # if game_nr % args.update_threshold == 0:
                #     avg_loss, batch_loss = process_data(batch)
                #     loss.append(avg_loss)
                #     avg_loss = "{:.2E}".format(Decimal(avg_loss))
                #     batch_loss = "{:.2E}".format(Decimal(batch_loss))
                #     batch.clear()

                # Update after X frames
                batch_loss = sum([avg * args.batch_size for avg in loss_list])
                avg_loss = sum(loss_list) / len(loss_list)
                loss_list.clear()

                # train_data = [batch[randint(0, len(batch) - 1)] for _ in range(args.batch_size)]
                # avg_loss, batch_loss = process_data(train_data)
                losses.append(avg_loss)
                avg_loss = "{:.2E}".format(Decimal(avg_loss))
                batch_loss = "{:.2E}".format(Decimal(batch_loss))
                tqdm.write(" {:5d} | {:10.8f} | {:+5.0f} | {:10.5f} | {:10.5f} | {:12d} | {:^10} | {:^15}".format(
                    game_nr, eps_threshold, running_reward, sum(rewards)/len(rewards),
                    sum(rewards[-10:])/len(rewards[-10:]),
                    moves, avg_loss, batch_loss
                    )
                )
                break

    summary(rewards, losses)
    serializers.save_hdf5(args.outfile if args.outfile else args.env, net)


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
    parser.add_argument("--epochs", dest="n_epoch", type=int, default=20000,
                        help="Amount of epochs")
    parser.add_argument("--replay-size", dest="replay_size", type=int, default=100000,
                        help="Size of replay buffer")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128,
                        help="Size of replay buffer")
    parser.add_argument("--alpha", dest="alpha", type=float, default=1e-4,
                        help="Learning Rate")
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.05,
                        help="Chance of doing a random action")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--decay-rate", dest="decay_rate", type=float, default=0.99,
                        help="Decay rate for future rewards")
    parser.add_argument("--update-after", dest="update_threshold", type=int, default=10,
                        help="Number of frames needed to update")
    parser.add_argument("--headless", type=str2bool, nargs='?', const=True, default=False,
                        help="Headless mode, suppresses rendering and plotting")
    args = parser.parse_args()

    env = gym.make("Pong-v0")

    epsilon_max = 0.9
    epsilon_min = args.epsilon
    epsilon_decay = 100000
    eps_threshold = 1
    total_moves = 0

    net = FCN(n_actions=env.action_space.n)
    if chainer.cuda.available:
        net.to_gpu()
    optim = optimizers.RMSprop(lr=args.alpha)
    optim.setup(net)

    train()




