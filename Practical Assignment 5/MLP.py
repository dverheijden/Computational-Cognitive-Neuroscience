from chainer import Chain
import chainer.functions as F
import chainer.links as L
import numpy as np


class MLP(Chain):
    """
        Neural Network Definition, Multilayer Perceptron
        l1: fully connected hidden layer
        l2: output layer
    """
    def __init__(self, n_hidden, n_action):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)   # input layer -> hidden layer
            self.l2 = L.Linear(None, n_action)  # hidden layer -> output layer

    def __call__(self, observation):
        """
        Feed the data in a forward fashion through the MLP
        :param x: Data
        :return: Last Link of the MLP
        """
        layer_hidden = F.relu(self.l1(np.array([observation])))
        layer_output = self.l2(layer_hidden)
        return layer_output
