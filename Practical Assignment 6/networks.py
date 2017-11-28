from chainer import Chain
import chainer.links as L
import chainer.functions as F


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


class Generative(Chain):
	def __init__(self, n_hidden):
		super(Generative, self).__init__()
		with self.init_scope():
			self.fc = L.Linear(None, n_hidden)  # Fully Connected Layer
			self.bn = L.BatchNormalization(n_hidden)  # Batch Normalisation Computation
			self.deconv = L.Deconvolution2D(None, 28*28, 4, 2, 1)  # Deconvolutional Layer

	def __call__(self, x):
		fc_output = self.fc(x)
		batch_norm = self.bn(fc_output)
		deconv_output = F.sigmoid(self.deconv(batch_norm))

		return deconv_output


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


class Discriminative(Chain):
	def __init__(self, n_feature_maps=5, ksize=5):
		super(Discriminative, self).__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(1, n_feature_maps, ksize=ksize)  # Convolutional Layer
			self.lin = L.Linear(None, 1)  # Linear Readout Layer

	def __call__(self, x):
		conv_output = F.relu(self.conv(x))
		lin_output = self.lin(conv_output)

		return lin_output
