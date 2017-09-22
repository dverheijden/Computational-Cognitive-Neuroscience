from chainer import Chain
import chainer.functions as F
import chainer.links as L


class FullyConnectedNet(Chain):
    """
        Neural Network Definition
        Consists of N layers with 20 hidden units in total.
        Each hidden layer has 20 / (N - 1) units
        If 20 % (20 / (N - 1)) != 0, the remaining units are added to the first layer
    """

    total_units = 20
    layers = []

    def __init__(self, N, n_classes):
        super(FullyConnectedNet, self).__init__()
        with self.init_scope():
            self.N = N
            if N == 1:
                self.layers.append(L.Linear(None, n_classes))
            elif N == 2:
                n_hidden = int(self.total_units / (N - 1))
                self.layers.append(L.Linear(None, n_hidden))
                self.layers.append(L.Linear(None, n_classes))
            else:
                #  Calculate units per layer rounded down
                n_hidden = int(self.total_units / (N - 1))
                for _ in range(N-1):
                    self.layers.append(L.Linear(None, n_hidden))
                self.layers.append(L.Linear(None, n_classes))

                n_remaining = self.total_units - n_hidden * (N-1)
                if n_remaining > 0:
                    self.layers[0] = L.Linear(None, n_hidden + n_remaining)

    def __call__(self, x):
        """
        Feed the data in a forward fashion through the MLP
        :param x: Data
        :return: Last Link of the MLP
        """
        if self.N == 1:
            return self.layers[0](x)

        if self.N == 2:
            layer_hidden = F.relu(self.layers[0](x))
            return self.layers[1](layer_hidden)

        layer_outputs = []
        layer_outputs.append(F.relu(self.layers[0](x)))
        for i in range(1, self.N-1):
            layer_outputs.append(F.relu(self.layers[i](layer_outputs[-1])))
        layer_output = self.layers[-1](layer_outputs[-1])
        return layer_output




class Convolutional(Chain):
    """
        Neural Network Definition, Of a Convo
        l1: fully connected hidden layer
        l2: output layer
    """
    def __init__(self):
        super(Convolutional, self).__init__()
        with self.init_scope():
            feature_maps = 1
            self.convLayer = L.Convolution2D(None, feature_maps, ksize=5)
            self.mPoolLayer = L.Convolution2D(feature_maps, 5, ksize=2)
            self.fullyConnectedOutput = L.Linear(None, 10)


    def __call__(self, x):
        """
        Feed the data in a forward fashion through the MLP
        :param x: Data
        :return: Last Link of the MLP
        """
        print('ndim = ' + str(x.ndim)) 
        conv_units = (self.convLayer(x))
        # max_pooling_units = F.max_pooling_2d(self.mPoolLayer(conv_units), ksize=2)
        # connected = self.fullyConnectedOutput(x)
        # return connected
