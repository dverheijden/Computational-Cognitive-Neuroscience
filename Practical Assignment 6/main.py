import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import networks
from Model import Model
from utils import get_mnist, RandomIterator


def lossfun_generative(pred_discriminator, t):
	"""
	Loss has to be minimized, train generative model to fool discriminator
	i.e. let discriminator predict ones
	:param pred_discriminator:
	:return:
	"""
	return sum(F.log(t - pred_discriminator))


def train():
	loss_disc = []
	loss_gen = []
	for _ in tqdm(range(n_iter)):
		loss_disc_current = 0
		loss_gen_current = 0
		for data in train_iter:
			# Train D on real data
			x_real = data[0]
			t_real = np.ones(shape=(batch_size, 1), dtype=np.int32)

			discriminative_optimizer.update(discriminative_model, x_real, t_real)
			loss_disc_current += discriminative_model.loss

			# Train D on fake data
			gen_input = np.float32(np.random.uniform(size=(batch_size, 1)))
			x_fake = generative_model.predict(gen_input)
			t_fake = np.zeros(shape=(batch_size, 1), dtype=np.int32)

			discriminative_optimizer.update(discriminative_model, x_fake, t_fake)
			loss_disc_current += discriminative_model.loss

			# Train G
			predictions = discriminative_model.y
			generative_optimizer.update(generative_model, gen_input, np.ones(shape=(batch_size, 1), dtype=np.int32))
			loss_gen_current += generative_model.loss

		loss_disc.append(loss_disc_current.data/(2*train_iter.idx))
		loss_gen.append(loss_gen_current.data/train_iter.idx)

	gen_input = np.float32(np.random.uniform(size=[1, 1]))
	generation = generative_model.predict(gen_input)  # we need to keep the variable type around, to compute stuff

	# plt.imshow(np.reshape(generation.data, newshape=[28, 28]).transpose())
	plt.plot(loss_disc)
	plt.plot(loss_gen)
	plt.show()


if __name__ == "__main__":
	n_iter = 20
	batch_size = 50
	train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=True, classes=[0], n_dim=3)
	train_iter = RandomIterator(train_data, batch_size)
	test_iter = RandomIterator(test_data, batch_size)

	# discriminative_net = networks.DiscriminativeMLP(n_hidden=20)
	# generative_net = networks.GenerativeMLP(n_hidden=20)

	discriminative_net = networks.Discriminative()
	generative_net = networks.Generative(64)

	discriminative_model = Model(discriminative_net, lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)
	generative_model = Model(generative_net, lossfun=F.sigmoid_cross_entropy, accfun=None)

	discriminative_optimizer = optimizers.SGD()
	discriminative_optimizer.setup(discriminative_model)
	generative_optimizer = optimizers.SGD()
	generative_optimizer.setup(generative_model)

	train()
