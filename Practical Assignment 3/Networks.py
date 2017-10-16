from chainer import Chain
import chainer.links as L
from chainer import Variable
import numpy as np


class RNN(Chain):
	def __init__(self, n_hidden):
		super(RNN, self).__init__()
		with self.init_scope():
			self.lstm = L.LSTM(None, n_hidden)  # LSTM layer
			self.out = L.Linear(n_hidden, 1)  # feed-forward output layer

	def reset_state(self):
		self.lstm.reset_state()

	def __call__(self, x):
		# transform the input 'x' to a compatible input for an LSTM layer
		x = Variable(np.array([np.float32([x])]))

		l1 = self.lstm(x)
		y = self.out(l1)

		return y
