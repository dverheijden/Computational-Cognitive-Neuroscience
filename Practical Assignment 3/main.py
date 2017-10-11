import numpy as np
import chainer.links as L
from chainer import optimizers
from chainer.datasets import TupleDataset
from Networks import RNN
from Regressor import Regressor


# create toy data - compute sum of the previous and current input
def create_data(n=3000):
	X = np.random.rand(n, 1).astype('float32')
	T = np.sum(np.hstack((X[0:-1], X[1:])), axis=1)
	T = np.hstack([0, T[0:]]).astype('float32')
	T = T.reshape([n, 1])

	return TupleDataset(X, T)


def compute_loss(data):
	loss = 0

	print(data[1][0])
	loss += model(data[0][0], data[1][0])

	print('Loss: {}'.format(loss))
	return loss


def feed_data():
	for data in dataset:
		rnn.reset_state()
		optimizer.update(compute_loss, data)


if __name__ == "__main__":
	dataset = create_data()

	print(dataset.__getitem__(0)[0])
	print(dataset.__getitem__(1))

	rnn = RNN(n_hidden=50)

	model = Regressor(rnn)

	optimizer = optimizers.SGD()
	optimizer.setup(rnn)

	feed_data()
