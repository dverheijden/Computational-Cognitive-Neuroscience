from utils import get_mnist
import networks
from Model import Model
import chainer.functions as F
import chainer.optimizers as optimizers

if __name__ == "__main__":
	train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=False, classes=[0])

	discriminative_net = networks.Discriminative()
	generative_net = networks.Generative(n_hidden=20)

	discriminative_model = Model(discriminative_net, lossfun=F.sigmoid_cross_entropy, accfun=F.accuracy)
	generative_model = Model(generative_net, lossfun=None, accfun=None)

	discriminative_optimizer = optimizers.SGD()
	discriminative_optimizer.setup(discriminative_model)
	generative_optimizer = optimizers.SGD()
	generative_optimizer.setup(generative_model)
