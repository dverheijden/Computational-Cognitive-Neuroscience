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
			test_loss, test_accuracy = feed_data(data)

			train_loss_list.append(train_loss)
			test_loss_list.append(test_loss)
			print('Epoch {} \n'
			      'Training: accuracy: {} \t loss: {} \n'
			      'Testing: accuracy: {} \t loss: {}'.format(epoch + 1,
			                                                 train_accuracy, train_loss,
			                                                 test_accuracy, test_loss))

	return [train_loss_list, test_loss_list]


def feed_data(data):

	total_loss = 0
	total_accuracy = 0
	for i, label in data:
		optimizer.update(compute_loss, i, label)
		total_loss += float(model.loss.data)
		total_accuracy += float(model.accuracy.data)
	return total_loss / len(data), total_accuracy / len(data)

def with_trainer():
	epochs = 100

	train = create_data()
	test = create_data()

	train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=False)
	test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

	rnn = RNN(n_hidden=50)

	model = Regressor(rnn, compute_loss)

	# Set up the optimizer
	optimizer = optimizers.SGD()
	optimizer.setup(rnn)

	# Set up the trainer
	updater = training.StandardUpdater(train_iter, optimizer)
	trainer = training.Trainer(updater, (epochs, 'epoch'))

	# Evaluate the model with the test dataset for each epoch
	trainer.extend(extensions.Evaluator(test_iter, model))

	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
	trainer.extend(extensions.ProgressBar())

	trainer.run()


if __name__ == "__main__":
	n_epochs = 5

	train = create_data()
	test = create_data()

	train_iter = StreamingIterator(train, batch_size=3000)
	test_iter = StreamingIterator(test, batch_size=200)

	rnn = RNN(n_hidden=5)

	model = Regressor_v0(rnn)

	# Set up the optimizer
	optimizer = optimizers.SGD()
	optimizer.setup(model)

	train_network()
