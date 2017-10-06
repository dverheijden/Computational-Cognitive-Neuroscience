import numpy as np
import chainer.links as L
from chainer import optimizers
from chainer.datasets import TupleDataset
from Networks import RNN
from Regressor import Regressor
from chainer import Chain
import chainer.functions as F
from chainer import Variable

import chainer.links as L

l = L.LSTM(100, 50)
out = L.Linear(50, 1)

l.reset_state()
x = Variable(np.random.randn(10, 100).astype(np.float32))

y = l(x)
y = out(y)

l.reset_state()
x2 = Variable(np.random.randn(10, 100).astype(np.float32))
y2 = l(x)

print("x1: {}, x2: {}".format(y, y2))


