from chainer import Chain
import chainer.functions as F
import chainer.links as L


class FullyConnectedNet(Chain):
    """
        Neural Network Definition
    """

    layers = []
    n_hidden = 10

    def __init__(self, N, n_classes):
        super(FullyConnectedNet, self).__init__()
        with self.init_scope():
            self.N = N
            if N == 1:
                self.l1 = L.Linear(None, n_classes)
            elif N == 2:
                self.l1 = L.Linear(None, self.n_hidden)  # input layer -> hidden layer
                self.l2 = L.Linear(None, n_classes)  # hidden layer -> output layer
            elif N == 3:
                self.l1 = L.Linear(None, self.n_hidden)  # input layer -> hidden layer1
                self.l2 = L.Linear(None, self.n_hidden)  # hidden layer1 -> hidden layer 2
                self.l3 = L.Linear(None, n_classes)  # hidden layer2 -> output layer

    def __call__(self, x):
        """
        Feed the data in a forward fashion through the MLP
        :param x: Data
        :return: Last Link of the MLP
        """
        layer_output = None
        if self.N == 1:
            layer_output = self.l1(x)

        if self.N == 2:
            layer_hidden = F.relu(self.l1(x))
            layer_output = self.l2(layer_hidden)

        if self.N == 3:
            layer_hidden = F.relu(self.l1(x))
            layer_hidden2 = F.relu(self.l2(layer_hidden))
            layer_output = self.l2(layer_hidden2)

        return layer_output
