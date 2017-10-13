import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.datasets import TupleDataset
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from Networks import RNN
from Regressor import Regressor

# create toy data - compute sum of the previous and current input
from Regressor_v0 import Regressor_v0
from utils import StreamingIterator
import matplotlib.pyplot as plt


def create_data(n=3000):
	X = np.random.rand(n, 1).astype('float32')
	T = np.sum(np.hstack((X[0:-1], X[1:])), axis=1)
	T = np.hstack([0, T[0:]]).astype('float32')
	T = T.reshape([n, 1])

	return TupleDataset(X, T)


def compute_loss(x, label):
	loss = 0

	loss += model(x, label)

	return loss


def train_network():
	"""
	Trains the network for n_epochs.
	One epoch contains of a training phase and testing phase. Afterwards, the results are printed to the screen
	"""

	train_loss_list = []
	test_loss_list = []

	for epoch in range(n_epochs):
		rnn.reset_state()
		for data in train_iter:
			train_loss, train_accuracy = feed_data(data)
			# test_loss, test_accuracy = feed_data(data)

			train_loss_list.append(train_loss)
			# test_loss_list.append(test_loss)
		print('Epoch {} \n'
				'Training: accuracy: {} \t loss: {} \n'
				# 'Testing: accuracy: {} \t loss: {}'
				.format(epoch + 1,
															train_accuracy, train_loss,
															# test_accuracy, test_loss
															))

	return [train_loss_list, test_loss_list]


def feed_data(data):

	total_loss = 0
	total_accuracy = 0
	for i, label in data:
		optimizer.update(compute_loss, i, label)
		total_loss += float(model.loss.data)
		total_accuracy += float(model.accuracy.data)
	print(total_loss/len(data))
	return total_loss / len(data), total_accuracy / len(data)


if __name__ == "__main__":
	n_epochs = 5

	train = create_data(1000)
	test = create_data(20)

	train_iter = StreamingIterator(train, batch_size=100)
	test_iter = StreamingIterator(test, batch_size=5)

	rnn = RNN(n_hidden=30)

	model = Regressor_v0(rnn)

	# Set up the optimizer
	optimizer = optimizers.SGD()
	optimizer.setup(model)


	train_network()


	graph_zise = 100
	test = create_data(graph_zise)
	rnn.reset_state()
	result_list = []
	expected_list = []
	for x, label in test:
		result_list.append(rnn(x).data[0,0])
		expected_list.append(label[0])
	
	plt.plot(np.arange(graph_zise), result_list, 'bs', np.arange(graph_zise), expected_list, 'g^')
	plt.show()