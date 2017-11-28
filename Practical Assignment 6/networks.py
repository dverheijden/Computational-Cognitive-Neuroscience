from chainer import Chain
import chainer.links as L
import chainer.functions as F
from functools import reduce


class GenerativeMLP(Chain):
	def __init__(self, n_hidden):
		super(GenerativeMLP, self).__init__()
		with self.init_scope():
			self.fc1 = L.Linear(None, n_hidden)  # Fully Connected Layer
			self.fc2 = L.Linear(n_hidden, n_hidden)
			self.fc3 = L.Linear(n_hidden, 28*28)

	def __call__(self, x):
		fc1_out = F.elu(self.fc1(x))
		fc2_out = F.sigmoid(self.fc2(fc1_out))
		fc3_out = self.fc3(fc2_out)
		return fc3_out


class DiscriminativeMLP(Chain):
	def __init__(self, n_hidden):
		super(DiscriminativeMLP, self).__init__()
		with self.init_scope():
			self.fc1 = L.Linear(784, n_hidden)  # Fully Connected Layer
			self.fc2 = L.Linear(n_hidden, n_hidden)
			self.fc3 = L.Linear(n_hidden, 2)

	def __call__(self, x):
		fc1_out = F.elu(self.fc1(x))
		fc2_out = F.elu(self.fc2(fc1_out))
		fc3_out = F.sigmoid(self.fc3(fc2_out))
		return fc3_out


def lindim(dims, scale, n):
	d = map(lambda x: x // scale, dims)
	d = reduce(lambda x, y: x * y, d)
	return d * n


def convdim(dims, scale, n):
	return n, dims[0] // scale, dims[1] // scale


class Generative(Chain):
	def __init__(self, n_hidden):
		super(Generative, self).__init__()
		with self.init_scope():
			self.n_hidden = n_hidden
			self.fc0 = L.Linear(None, n_hidden)  # Fully Connected Layer
			self.deconv1 = L.Deconvolution2D(n_hidden, 128, 4, stride=2, pad=1)  # Deconvolutional Layer
			self.deconv2 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1)  # Deconvolutional Layer
			self.deconv3 = L.Deconvolution2D(64, 1, 24, stride=2, pad=1)  # Deconvolutional Layer
			self.bn0 = L.BatchNormalization(n_hidden)  # Batch Normalisation Computation
			self.bn1 = L.BatchNormalization(128)  # Batch Normalisation Computation
			self.bn2 = L.BatchNormalization(64)  # Batch Normalisation Computation

	def __call__(self, x):
		h = F.relu(self.bn0(self.fc0(x)))
		h = F.reshape(h, (x.shape[0],) + (self.n_hidden, 1, 1))
		h = F.relu(self.bn1(self.deconv1(h)))
		h = F.relu(self.bn2(self.deconv2(h)))
		h = F.sigmoid(self.deconv3(h))

		return h


class Discriminative(Chain):
	def __init__(self, n_feature_maps=5):
		super(Discriminative, self).__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(None, n_feature_maps, ksize=5)  # Convolutional Layer
			self.lin = L.Linear(None, 2)  # Linear Readout Layer

	def __call__(self, x):
		conv_output = F.relu(self.conv(x))
		max_pooling_output = F.max_pooling_2d(conv_output, ksize=2)
		lin_output = self.lin(max_pooling_output)

		return lin_output
