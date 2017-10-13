import numpy as np
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
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

	loss += model(data[0][0], data[1][0])

	return loss


def feed_data():
	for data in dataset:
		rnn.reset_state()
		optimizer.update(compute_loss, data)


if __name__ == "__main__":
	epochs = 100


	dataset = create_data()

	rnn = RNN(n_hidden=50)

	model = Regressor(rnn)

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
