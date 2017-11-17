from utils import get_mnist

train_data, test_data = get_mnist(n_train=1000, n_test=100, with_label=False, classes=[0])
