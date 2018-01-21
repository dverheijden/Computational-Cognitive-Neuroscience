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
from random import shuffle, randint
from decimal import Decimal


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
    new_data = []
    for cur_state, next_state, old_q, taken_action, reward, terminal in data:
        new_q = old_q.data.copy()

        _, _, max_next_Q = compute_action(next_state, deterministic=True)

        # Simple DQN approach
        new_q[0][taken_action] = reward + args.gamma * max_next_Q if not terminal else reward

        # Old Approach
        # new_q[0][taken_action] += args.alpha * (reward + args.gamma * max_next_Q - new_q[0][taken_action])
        new_q = Variable(new_q)
        new_data.append([old_q, new_q])

    # shuffle(new_data) only needed without replay memory

    cumul_loss = 0
    for old_q, new_q in new_data:
        loss_prog = F.mean_squared_error(old_q, new_q)
        prog_net.cleargrads()
        loss_prog.backward()
        prog_optimizer.update()
        cumul_loss += loss_prog.data

    # tqdm.write("Trained on {} samples \t Loss: {}".format(len(data), cumul_loss))
    return float(cumul_loss / len(data)), float(cumul_loss)


def discount_reward(memory, cur_reward):
    discounted_reward = np.zeros(len(memory))
    discounted_reward[-1] = cur_reward
    for t in reversed(range(0, len(memory) - 1)):
        discounted_reward[t] = args.decay_rate * discounted_reward[t+1]

    return discounted_reward


def preprocess_obs(obs):
    """

    :param obs: (210,160,3) field
    :return:
    """
    # TODO: Maybe trim the unessential parts out of the obs
    # TODO: Turn obs to grayscale?
    obs = np.array(obs)  # Convert to potential cupy array
    obs = obs.astype('float32')
    obs = obs.transpose((2, 0, 1))
    obs = np.expand_dims(obs, axis=0)
    if args.env == "Pong-v0":
        obs = obs[35:195, :, :]  # Trim out score and bottom
        obs = obs[::2, ::2, :]  # Downsample by a factor 2
        # obs = obs.reshape((1, 3, 80, 80))  # Chainer needs (n_samples, n_channels, length, width)
    # else:
        # obs = obs.reshape((1, 3, 210, 160))

    return obs


def train():
    print("observation space:", env.observation_space)
    print("action space:", env.action_space)

    rewards = []
    losses = []
    tqdm.write(" {:^5} | {:^5} | {:^10} | {:^10} | {:^12} | {:^10} | {:^15}\n".format(
        "Game", "Score", "Total Avg", "Score@10", "Total Moves", "Avg Loss", "Total Loss")
               + "-"*88)
    with open("{}.progress".format(args.env), 'w') as f:
        for game_nr in tqdm(range(1, args.n_epoch+1), unit="game", ascii=True, file=f):
            prev_obs = None
            cur_obs = preprocess_obs(env.reset())

            batch = []
            # memory = []

            running_reward = 0

            moves = 0

            loss_list = []

            while True:
                moves += 1

                if not args.headless:
                    env.render()

                obs = cur_obs - prev_obs if prev_obs is not None else np.zeros(cur_obs.shape)
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
                if len(batch) is args.replay_size:
                    batch[randint(0, len(batch)-1)] = [prev_obs, cur_obs, q_values, taken_action, reward, done]
                else:
                    batch.append([prev_obs, cur_obs, q_values, taken_action, reward, done])

                # Update after every args.update_threshold frames
                if moves % args.update_threshold is 0:
                    train_data = [batch[randint(0, len(batch) - 1)] for _ in range(args.batch_size)]
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
                    tqdm.write(" {:5d} | {:+5.0f} | {:10.5f} | {:10.5f} | {:12d} | {:^10} | {:^15}".format(
                        game_nr, running_reward, sum(rewards)/len(rewards),
                        sum(rewards[-10:])/len(rewards[-10:]),
                        moves, avg_loss, batch_loss
                        )
                    )
                    break

    serializers.save_hdf5(args.outfile, prog_net)
    summary(rewards, losses)


def compute_action(obs, deterministic):
    """
    Computes the next action in an e-greedy fashion
    :param obs:
    :param deterministic:
    :return: 
    """
    q_values = prog_net(obs, 1)

    if np.random.rand() < args.epsilon and not deterministic:
        taken_action = env.action_space.sample()
    else:
        if cuda.available:
            taken_action = 1
            max_Q = -99999999
            for index, Q in enumerate(q_values.data[0]):
                if Q > max_Q:
                    taken_action = index
                    max_Q = Q
        else:
            taken_action = np.argmax(q_values.data[0])

    return q_values, taken_action, q_values.data[0][taken_action]


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
    parser.add_argument("--epochs", dest="n_epoch", type=int, default=100,
                        help="Amount of epochs")
    parser.add_argument("--replay-size", dest="replay_size", type=int, default=1000000,
                        help="Size of replay buffer")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32,
                        help="Size of replay buffer")
    parser.add_argument("--alpha", dest="alpha", type=float, default=1e-4,
                        help="Learning Rate")
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1,
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
    print(args)
    if cuda.available:  # GPU optimization
        import cupy as np
        np.cuda.Device(0).use()
        mpl.use('Agg')  # Suppress rendering

    import matplotlib.pyplot as plt
    env = gym.make(args.env)

    number_of_actions = env.action_space.n

    if args.model_path:
        prog_net = run_saved(args.model_path)
    else:
        prog_net = networks.ProgNet(n_actions=number_of_actions, n_feature_maps=args.n_feature_maps,
                                    n_hidden_units=args.n_hidden)

    if cuda.available:
        prog_net.to_gpu(0)

    # prog_optimizer = optimizers.Adam(alpha=args.alpha)
    prog_optimizer = optimizers.RMSprop(lr=0.01)
    prog_optimizer.setup(prog_net)

    print("Model Set Up!")

    train()
