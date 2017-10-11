import chainer.links as L
from chainer import optimizers
from chainer.datasets import TupleDataset
from Networks import RNN
from Regressor import Regressor
from chainer import Chain
import chainer.functions as F
from chainer import Variable
import numpy as np


import chainer.links as L

l = L.LSTM(1, 50)
out = L.Linear(50, 1)

l.reset_state()
x = np.random.randn(1, 1).astype(np.float32)
x2 = Variable(x)

z1 = [np.float32(1)]
z2 = np.array([z1])
z3 = Variable(z2)

y = l(x2)
y = out(y)

l.reset_state()
y2 = l(z3)

print("x1: {}, x2: {}".format(y, y2))


