import chainer.functions as F
import chainer.optimizers as optimizers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import networks
from Model import Model
from utils import get_mnist, RandomIterator


def train():
	loss_disc = []
	loss_gen = []
	for _ in tqdm(range(n_iter)):
		loss_disc_current = 0
		loss_gen_current = 0
		for data in train_iter:
		
			x_real = data
			t_real = discriminative_net(x_real)

			gen_input = np.float32(np.random.uniform(size=(batch_size, 1)))
			x_fake = generative_net(gen_input)
			t_fake = discriminative_net(x_fake)

			# generative_loss = F.log(t_fake)
			

			# Backprop
			generative_loss = F.softmax_cross_entropy(t_fake, np.ones(shape=(batch_size), dtype=np.int32))
			discriminative_loss = F.softmax_cross_entropy(t_fake, np.zeros(shape=(batch_size), dtype=np.int32))
			discriminative_loss += F.softmax_cross_entropy(t_real, np.ones(shape=(batch_size), dtype=np.int32))
			discriminative_loss /= 2
			
			generative_net.cleargrads()
			generative_loss.backward()  # recompute the grads
			generative_optimizer.update()

			discriminative_net.cleargrads()
			discriminative_loss.backward()
			discriminative_optimizer.update()

			loss_disc_current += discriminative_loss.data
			loss_gen_current += generative_loss.data

		loss_gen.append(loss_gen_current/train_iter.idx)
		loss_disc.append(loss_disc_current/train_iter.idx)

	plt.plot(loss_disc, label="Discriminator Loss")
	plt.plot(loss_gen, label="Generator Loss")
	plt.legend()
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	plt.show()
	for i in range(4):
		gen_input = np.float32(np.random.uniform(size=[1, 1]))
		generation = generative_net(gen_input)  # we need to keep the variable type around, to compute stuff

		plt.imshow(np.reshape(generation.data, newshape=[28, 28]).transpose())
		plt.show()


if __name__ == "__main__":
	n_iter = 100
	batch_size = 50
	train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=False, classes=[0], n_dim=3)
	train_iter = RandomIterator(train_data, batch_size)
	test_iter = RandomIterator(test_data, batch_size)

	# discriminative_net = networks.DiscriminativeMLP(n_hidden=20)
	# generative_net = networks.GenerativeMLP(n_hidden=200)

	discriminative_net = networks.Discriminative()
	generative_net = networks.Generative(256)

	discriminative_optimizer = optimizers.SGD()
	discriminative_optimizer.setup(discriminative_net)

	generative_optimizer = optimizers.SGD()
	generative_optimizer.setup(generative_net)

	train()
