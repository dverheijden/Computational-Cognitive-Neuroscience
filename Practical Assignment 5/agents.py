import numpy as np
from MLP import MLP
from Regressor import Regressor
from chainer import optimizers
from chainer import functions as F


class RandomAgent(object):
    def __init__(self, env):
        """
        Args:
        env: an environment
        """

        self.env = env

    def act(self, observation):
        """
        Act based on observation and train agent on cumulated reward (return)
        :param observation: new observation
        :param reward: reward gained from previous action; None indicates no reward because of initial state
        :return: action (Variable)
        """

        return np.random.choice(self.env.n_action)

    def train(self, a, old_obs, r, new_obs):
        """
        :param a: action
        :param old_obs: old observation
        :param r: reward
        :param new_obs: new observation
        :return:
        """

        pass


class TabularQAgent(object):
    def __init__(self, env, alpha=0.1, gamma=0.3):
        """
        Args:
            env: an environment
        """

        self.env = env
        self.Q = np.zeros((env.n_action**env.n_input, env.n_action)).astype(np.float32)
        self.alpha = alpha
        self.gamma = gamma

    def act(self, observation):
        """
        Act based on observation and train agent on accumulated reward (return)
        :param observation: new observation
        :param reward: reward gained from previous action; None indicates no reward because of initial state
        :return: action (Variable)
        """
        return 0 if self.Q[self.env.asint(observation), 0] > self.Q[self.env.asint(observation), 1] else 1

    def train(self, a, old_obs, r, new_obs):
        """
        :param a: action
        :param old_obs: old observation
        :param r: reward
        :param new_obs: new observation
        :return:
        """
        old_obs = self.env.asint(old_obs)
        new_obs = self.env.asint(new_obs)

        max_Q = self.Q[new_obs, np.argmax(self.Q[new_obs, :])]

        self.Q[old_obs, a] = (1-self.alpha)*self.Q[old_obs, a] + self.alpha*(r + self.gamma*max_Q)


class NeuralAgent(object):
    def __init__(self, env, actualQ):
        """
        Args:
        env: an environment
        """
        n_hidden = 10

        self.MLP = MLP(n_hidden, env.n_action)
        self.model = Regressor(self.MLP, lossfun=F.squared_error, accfun=None)
        self.env = env
        self.Q = np.zeros((env.n_action ** env.n_input, env.n_action)).astype(np.float32)
        self.actualQ = actualQ

        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model)

    def compute_loss(self, y, t):
        """
        We define the loss as the sum of the squared error between the actual and predicted Q values
        :param y: predicted Q-values
        :param t: actual Q-values
        :return: loss
        """
        return sum(np.square(np.subtract(y, t)))

    def act(self, observation):
        """
        Act based on observation and train agent on cumulated reward (return)
        :param observation: new observation
        :param reward: reward gained from previous action; None indicates no reward because of initial state
        :return: action
        """
        x = self.model.predictor(observation).data
        action = np.argmax(x)

        return action

    def train(self, a, old_obs, r, new_obs):
        """
        :param a: action
        :param old_obs: old observation
        :param r: reward
        :param new_obs: new observation
        :return:
        """

        newQ = self.model.predictor(old_obs).data

        _old_obs = self.env.asint(old_obs)
        self.Q[_old_obs, 0] = newQ[0, 0]
        self.Q[_old_obs, 1] = newQ[0, 1]

        self.optimizer.update(self.model.lossfun, self.Q[_old_obs, :], self.actualQ[_old_obs, :])

