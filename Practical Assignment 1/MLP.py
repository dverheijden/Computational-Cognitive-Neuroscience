import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class MLP(Chain):
    """
        Neural Network Definition, Multilayer Perceptron
        l1: hidden layer
        l2: output layer
    """
    def __init__(self, n_hidden, n_classes):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(None, n_classes)

    def __call__(self, x):
        layer_hidden = F.relu(self.l1(x))
        layer_output = self.l2(layer_hidden)
        return layer_output
