import numpy as np
from chainer import optimizers
from chainer.datasets import TupleDataset
from Networks import RNN
from Regressor import Regressor
import pickle

from tqdm import tqdm

from utils import StreamingIterator
import matplotlib.pyplot as plt


def create_data(n=3000):
	X = np.random.rand(n, 1).astype('float32')
	T = np.sum(np.hstack((X[0:-1], X[1:])), axis=1)
	T = np.hstack([0, T[0:]]).astype('float32')
	T = T.reshape([n, 1])

	return TupleDataset(X, T)


def compute_loss(y, t):
	"""
	We define the loss as the squared error between the predicted and actual sum
	:param y: predicted sum
	:param t: actual sum
	:return: loss
	"""
	return pow(y - t, 2)


def compute_accuracy(y, t):
	"""
	We define the accuracy as 1 over the distance between the predicted and actual sum
	:param y: predicted sum
	:param t: actual sum
	:return: accuracy
	"""
	return 1 / abs(y - t)


def train_network(sample_iter, rnn, model, optimizer):
	"""
	Trains the network for n_epochs.
	One epoch contains of a training phase and testing phase.
	:param sample_iter: training data iterator
	:param rnn:
	:param model:
	:param optimizer:
	:return: average test and training loss per epoch
	"""
	train_loss_list = []
	test_loss_list = []

	for epoch in range(n_epochs):
		# print('Epoch {}'.format(epoch + 1))
		current_train_loss = 0
		current_test_loss = 0
		for batch in tqdm(sample_iter, leave=False):
			rnn.reset_state()  # new batch -> reset states
			train_loss, train_accuracy = feed_data(batch, model, optimizer, True)
			test_loss, test_accuracy = feed_data(batch, model, optimizer, False)

			current_train_loss += train_loss
			current_test_loss += test_loss
			# print("\t Training: accuracy: {} \t loss: {} \n\t Testing: accuracy: {} \t loss: {}".format(
			# 	train_accuracy,
			# 	train_loss,
			# 	test_accuracy,
			# 	test_loss))
		train_loss_list.append(current_train_loss / len(sample_iter))
		test_loss_list.append((current_test_loss / len(sample_iter)))
	return [train_loss_list, test_loss_list]


def feed_data(batch, model, optimizer, update):
	"""
	Feed the data through the network and update the weights if needed
	:param batch: sequential data
	:param model:
	:param optimizer:
	:param update: bool on whether to update the weights
	:return: average loss and accuracy of the batch
	"""
	total_loss = 0
	total_accuracy = 0
	for x, t in batch:
		model(x, t)
		if update:
			optimizer.update(compute_loss, model.y, t)
		total_loss += float(model.loss.data)
		total_accuracy += float(model.accuracy.data)
	return total_loss / len(batch), total_accuracy / len(batch)


def test_network(sample_iter):
	"""
	Test the network by feeding it the sequential data
	Plot the predicted and actual sum per sample
	Plot the difference between the actual and predicted sum
	:param sample_iter: Test iterator object over test samples
	:return:
	"""
	rnn.reset_state()  # reset internal states
	predicted_sum = []
	actual_sum = []
	for sample in sample_iter:
		for data in sample:
			x, t = data
			model(x, t)
			actual_sum.append(t[0])  # unpack values
			predicted_sum.append(model.y.data[0][0])  # unpack values

	plt.plot(predicted_sum, 'ro', label="predicted sum")
	plt.plot(actual_sum, 'go', label="actual sum")
	plt.xlabel("sample")
	plt.ylabel("output")
	plt.legend()
	plt.show()

	difference = abs(np.subtract(predicted_sum, actual_sum))
	plt.plot(difference)
	plt.xlabel("sample")
	plt.ylabel("prediction error")
	plt.show()


def get_model(model_name):
	"""
	Load the named model if it exists, otherwise train it
	:param model_name: Name of the model
	:return: Network, Regressor, Optimizer and results
	"""
	try:
		pickle_in = open("{}_rnn.pickle".format(model_name), 'rb')
		rnn = pickle.load(pickle_in)

		pickle_in = open("{}_model.pickle".format(model_name), 'rb')
		model = pickle.load(pickle_in)

		pickle_in = open("{}_optimizer.pickle".format(model_name), 'rb')
		optimizer = pickle.load(pickle_in)

		pickle_in = open("{}_results.pickle".format(model_name), 'rb')
		results = pickle.load(pickle_in)

		tqdm.write("Model '{}' Loaded!".format(model_name))

	except FileNotFoundError:
		rnn = RNN(n_hidden=hidden_units)

		model = Regressor(rnn, accfun=compute_accuracy, lossfun=compute_loss)

		# Set up the optimizer
		optimizer = optimizers.SGD()
		optimizer.setup(model)

		tqdm.write("Model not found! Starting training ...")
		results = train_network(train_iter, rnn, model, optimizer)

		with open('{}_rnn.pickle'.format(model_name), 'wb') as f:
			pickle.dump(rnn, f)
		with open('{}_model.pickle'.format(model_name), 'wb') as f:
			pickle.dump(model, f)
		with open('{}_optimizer.pickle'.format(model_name), 'wb') as f:
			pickle.dump(optimizer, f)
		with open('{}_results.pickle'.format(model_name), 'wb') as f:
			pickle.dump(results, f)

	# Plot the training and test loss as a function of epochs
	plt.plot(results[0], label='train loss')
	plt.plot(results[1], label='test loss')
	plt.legend()
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.show()
	return rnn, model, optimizer, results


if __name__ == "__main__":
	n_epochs = 25
	batch_size = 100
	train_data_size = 3000
	hidden_units = 50

	train_data = create_data(n=train_data_size)
	test_data = create_data(n=100)

	train_iter = StreamingIterator(train_data, batch_size=batch_size)
	test_iter = StreamingIterator(test_data, batch_size=len(test_data))

	rnn, model, optimizer, results = get_model("n_epochs{}_batch_size{}_train_size{}_hidden_units{}".format(
		n_epochs,
		batch_size,
		train_data_size,
		hidden_units
	))

	test_network(test_iter)
