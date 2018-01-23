import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib as mpl
from chainer import serializers
from chainer import Variable
import chainer
import numpy as np
from tqdm import tqdm
import gym
from networks import FCN, CNN
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
    """
    Process data by computing their new Q (target) values, compute the (Huber) loss and backprop
    We circumvent the use of a target network by pre-computing the targets
    :param data:
    :return:
    """
    cumul_loss = 0
    backprop_temp = []
    for cur_state, next_state, old_q, taken_action, terminal, reward in data:

        with chainer.no_backprop_mode():
            new_q = old_q.data[0].copy()
            new_q = chainer.cuda.to_cpu(new_q) if chainer.cuda.available else new_q

            _, _, max_next_Q = compute_action(next_state, deterministic=True)

            new_q[taken_action] = reward + args.gamma * max_next_Q.data if not terminal else reward
            new_q = np.expand_dims(new_q, axis=0)
            backprop_temp.append([old_q, new_q])

    for old_q, new_q in backprop_temp:
        new_q = chainer.cuda.to_gpu(new_q) if chainer.cuda.available else new_q
        loss = F.huber_loss(old_q, Variable(new_q), 1)
        net.cleargrads()
        loss.backward()
        optim.update()
        cumul_loss += loss.data

    # tqdm.write("Trained on {} samples \t Loss: {}".format(len(data), cumul_loss))
    return float(cumul_loss / len(data)), float(cumul_loss)


def preprocess_obs(obs, dim=2):
    """
    Process one observation to a 1D array
    :param obs:
    :return:
    """

    obs = obs.astype(np.float32)

    if args.env == "Pong-v0":
        obs = obs[35:195]  # crop
        obs = obs[::2, ::2, 0]  # downsample by factor of 2 and take only RED
        obs[obs == 144] = 0  # erase background (background type 1)
        obs[obs == 109] = 0  # erase background (background type 2)
        obs[obs != 0] = 1  # everything else (paddles, ball) just set to 1

        if dim == 2:
            obs = obs.ravel()
        elif dim == 3:
            obs = np.expand_dims(obs, axis=0)

    obs = np.expand_dims(obs, axis=0)
    return obs


def discount_reward(memory, cur_reward):
    discounted_reward = np.zeros(len(memory))
    discounted_reward[-1] = cur_reward
    for t in reversed(range(0, len(memory) - 1)):
        discounted_reward[t] = args.decay_rate * discounted_reward[t+1]

    return discounted_reward


def compute_action(obs, deterministic):
    """
    Computes the next action in an e-greedy fashion
    :param obs:
    :param deterministic:
    :return:
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
        "Game", "Epsilon", "Score", "Total Avg", "Score@10", "Total Moves", "Avg Loss", "Total Loss")
               + "-"*100)
    with open("{}.progress".format(args.env), 'w') as f:

        temp_memory = []
        replay_memory = []
        loss_list = []

        for game_nr in tqdm(range(1, args.n_epoch+1), unit="game", ascii=True, file=f):
            prev_obs = None
            cur_obs = preprocess_obs(env.reset(), dim=3)

            running_reward = 0

            moves = 0

            while True:
                moves += 1  # Bookkeeping
                total_moves += 1  # epsilon_decay

                if not args.headless:
                    env.render()

                if args.env != "CartPole-v0":
                    state = np.subtract(cur_obs, prev_obs) if prev_obs is not None else np.zeros(cur_obs.shape, dtype=np.float32)
                else:
                    state = cur_obs

                prev_obs = cur_obs

                q_values, taken_action, _ = compute_action(state, deterministic=False)

                cur_obs, reward, done, info = env.step(taken_action)
                cur_obs = preprocess_obs(cur_obs, dim=3)

                if reward != 0:
                    reward = -1 if reward < 0 else 1

                if args.env != "CartPole-v0":
                    state_future = np.subtract(cur_obs, prev_obs)
                else:
                    state_future = cur_obs

                running_reward += reward

                # # Old Approach...
                # temp_memory.append([state, state_future, q_values, taken_action, done])
                #
                # if reward != 0 or done:
                #     # Distribute discounted reward to previous actions
                #     discounted_reward = discount_reward(temp_memory, reward)
                #     for i in range(len(temp_memory)):
                #         temp_memory[i].extend([discounted_reward[i]])
                #     replay_memory.extend(temp_memory)  # append to training data
                #     temp_memory.clear()
                #     while len(replay_memory) > args.replay_size:
                #         del replay_memory[np.random.randint(0, len(replay_memory))]

                # Simple DQN approach - Pin it on Value Iteration / Reward Propagation
                if len(replay_memory) is args.replay_size:
                    replay_memory[randint(0, len(replay_memory)-1)] = [state, state_future, q_values, taken_action, done, reward]
                else:
                    replay_memory.append([state, state_future, q_values, taken_action, done, reward])

                # Update after every args.update_threshold frames
                if moves % args.update_threshold is 0 and len(replay_memory) > args.batch_size:
                    train_data = [replay_memory[randint(0, len(replay_memory) - 1)] for _ in range(args.batch_size)]
                    avg_loss, _ = process_data(train_data)
                    loss_list.append(avg_loss)
                    losses.append(avg_loss)

                if done:
                    rewards.append(running_reward)
                    batch_loss = "N/A"
                    avg_loss = "N/A"
                    # # Old Approach
                    # if game_nr % args.update_threshold == 0:
                    #     shuffle(replay_memory)
                    #     avg_loss, batch_loss = process_data(replay_memory)
                    #     losses.append(avg_loss)
                    #     avg_loss = "{:.2E}".format(Decimal(avg_loss))
                    #     batch_loss = "{:.2E}".format(Decimal(batch_loss))

                    # Update after X frames Method
                    if len(loss_list) > 0:
                        batch_loss = sum([avg * args.batch_size for avg in loss_list])
                        avg_loss = sum(loss_list) / len(loss_list)
                        loss_list.clear()
                        avg_loss = "{:.2E}".format(Decimal(avg_loss))
                        batch_loss = "{:.2E}".format(Decimal(batch_loss))

                    # train_data = [batch[randint(0, len(batch) - 1)] for _ in range(args.batch_size)]
                    # avg_loss, batch_loss = process_data(train_data)
                    # losses.append(avg_loss)
                    if game_nr % args.plot_every is 0:
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
    parser.add_argument("--hidden", dest="n_hidden", type=int, default=512,
                        help="Amount of hidden units")
    parser.add_argument("--feature-maps", dest="n_feature_maps", type=int, default=12,
                        help="Amount of feature maps")
    parser.add_argument("--epochs", dest="n_epoch", type=int, default=20000,
                        help="Amount of epochs")
    parser.add_argument("--replay-size", dest="replay_size", type=int, default=1000000,
                        help="Size of replay buffer")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128,
                        help="Size of Batch Size")
    parser.add_argument("--alpha", dest="alpha", type=float, default=1e-5,
                        help="Learning Rate")
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.05,
                        help="Chance of doing a random action")
    parser.add_argument("--epsilon-decay", dest="epsilon_decay", type=float, default=200000,
                        help="Measure at which epsilon decays (very game dependent!)")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--decay-rate", dest="decay_rate", type=float, default=0.99,
                        help="Decay rate for future rewards")
    parser.add_argument("--update-after", dest="update_threshold", type=int, default=100,
                        help="Number of frames needed to update")
    parser.add_argument("--plot-every", dest="plot_every", type=int, default=1,
                        help="Number of games before showing summary")
    parser.add_argument("--headless", type=str2bool, nargs='?', const=True, default=False,
                        help="Headless mode, suppresses rendering and plotting")
    args = parser.parse_args()
    print(args)

    if args.headless:  # GPU optimization
        mpl.use('Agg')  # Suppress rendering

    import matplotlib.pyplot as plt

    env = gym.make(args.env)

    epsilon_max = 1
    epsilon_min = args.epsilon
    epsilon_decay = args.epsilon_decay
    eps_threshold = 1

    total_moves = 0

    net = FCN(n_actions=env.action_space.n)
    if chainer.cuda.available:
        net.to_gpu()
    optim = optimizers.RMSprop(lr=args.alpha)
    optim.setup(net)

    train()




