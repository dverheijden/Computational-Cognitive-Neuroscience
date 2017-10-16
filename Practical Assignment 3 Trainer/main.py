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
def create_data(n=3000):
	X = np.random.rand(n, 1).astype('float32')
	T = np.sum(np.hstack((X[0:-1], X[1:])), axis=1)
	T = np.hstack([0, T[0:]]).astype('float32')
	T = T.reshape([n, 1])

	return TupleDataset(X, T)


def compute_loss(y, t):
	return pow(y - t, 2)


def compute_accuracy(y, t):
	return abs(y - t)


if __name__ == "__main__":
	epochs = 10

	train = create_data(n=400)
	test = create_data(n=400)

	train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=False)
	test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

	rnn = RNN(n_hidden=50)

	model = Regressor(predictor=rnn, lossfun=compute_loss, accfun=compute_accuracy)

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
