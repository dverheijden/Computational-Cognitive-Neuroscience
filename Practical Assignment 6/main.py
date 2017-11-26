from utils import get_mnist, RandomIterator
import networks
from Model import Model
import chainer.functions as F
import chainer.optimizers as optimizers
import numpy as np
from tqdm import tqdm


def lossfun_generative(y, t):
	return F.log(1 - t)


def lossfun_discriminative_fake(y, t):
	return pow(y - t, 2)


def lossfun_discriminative_real(y, t):
	return pow(y - t, 2)

def train():
	for _ in tqdm(range(n_iter)):
		for data in train_iter:
			# Train D on real+fake
			x_real = data[0]
			t_real = data[1].reshape(batch_size,1) # to have the same shape as the predictions of the discriminator
			# t_real = np.ones([batch_size], dtype=np.int32)

			discriminative_optimizer.update(discriminative_model, x_real, t_real)

			# we need a random input, otherwise we will be generating the same number each time
			gen_input = np.float32(np.random.uniform(size=[batch_size, 1]))

			x_fake = generative_model.predictor(gen_input).data.reshape(batch_size,1,28,28) 
			t_fake = np.ones( shape=(batch_size,1,1), dtype=np.int32)

			predictions = np.empty(shape=(50,1))
			for i in range(batch_size): # make the 50 predictions:
				predictions[i] = discriminative_model(x_fake[i], t_fake[i]).data

			discriminative_optimizer.update() # weet niet zeker of dit werkt
			# now get the correctly and incorrectly classified numbers in order to train the generator
			# hoe kunnen we nu het generator model updaten aan de hand van de gediscriminate predicionts?

			# generative_optimizer.update()
		# Dit werkt!
		# print(discriminative_model.predictor(x_real))
		


if __name__ == "__main__":
	n_iter = 20
	batch_size = 50
	train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=True, classes=[0], n_dim=3)
	train_iter = RandomIterator(train_data, batch_size)
	test_iter = RandomIterator(test_data, batch_size)

	discriminative_net = networks.DiscriminativeMLP(n_hidden=20)
	generative_net = networks.GenerativeMLP(n_hidden=20)

	discriminative_model = Model(discriminative_net, lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)
	generative_model = Model(generative_net, lossfun=None, accfun=None)

	discriminative_optimizer = optimizers.SGD()
	discriminative_optimizer.setup(discriminative_model)
	generative_optimizer = optimizers.SGD()
	generative_optimizer.setup(generative_model)

	train()

