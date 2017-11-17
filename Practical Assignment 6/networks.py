from chainer import Chain
import chainer.links as L
import chainer.functions as F

class Generative(Chain):
	def __init__(self, n_hidden):
		super(Generative, self).__init__()
		with self.init_scope():
			self.fc = L.Linear(None, n_hidden)  # Fully Connected Layer
			self.deconv = L.Deconvolution2D(in_channels=n_hidden, out_channels=28*28)  # Deconvolutional Layer

	def __call__(self, x):
		fc_output = self.fc(x)
		batch_norm = L.BatchNormalization(fc_output)  # Batch Normalisation Computation
		deconv_output = F.sigmoid(self.deconv(batch_norm))

		return deconv_output


class Discriminative(Chain):
	def __init__(self, n_feature_maps=5, ksize=5):
		super(Discriminative, self).__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(None, n_feature_maps, ksize=ksize)  # Convolutional Layer
			self.lin = L.Linear(None, 1)  # Linear Readout Layer

	def __call__(self, x):
		conv_output = F.relu(self.conv(x))
		lin_output = self.lin(conv_output)

		return lin_output
