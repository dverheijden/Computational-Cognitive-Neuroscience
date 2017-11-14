import numpy as np
from MLP import MLP
from Regressor import Regressor
from chainer import optimizers


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
        self.Q = np.ndarray([env.n_action**env.n_input, env.n_action])
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
    def __init__(self, env):
        """
        Args:
        env: an environment
        """
        n_hidden = 10
        compute_accuracy = lambda y, t: 1 / abs(y - t)
        compute_loss = lambda y, t: pow(y - t, 2)
        self.MLP = MLP(n_hidden, env.n_action)
        self.model = Regressor(self.MLP, accfun=compute_accuracy, lossfun=compute_loss)
        self.env = env

        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model)

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

        label = not a if r == 0 else a

        self.model(old_obs, label)
        Q_max = Q_new[0,np.argmax(Q_new)]

        max_Q_new = Q_new[np.argmax(Q_new)]
        self.model(old_obs, r + max_Q_new)
         # MLP([0.0,1.0]) returnt iets van de vorm [0.123123, 0.123123] 
         # de Q waarden.
         # moeten we dan beide resultaten die worden verkregen updaten met old_obs?

        pass

