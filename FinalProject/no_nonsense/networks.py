import chainer.functions as F
import chainer.links as L
from chainer import Chain

"""
For the Atari experiments we used a model with 3 convolutional layers followed by a fully connected layer
and from which we predict the policy and value function. The convolutional layers are as follows. All have 12
feature maps. The first convolutional layer has a kernel of size 8x8 and a stride of 4x4. The second layer has a
kernel of size 4 and a stride of 2. The last convolutional layer has size 3x4 with a stride of 1. The fully connected
layer has 256 hidden units.
"""


class CNN(Chain):
    def __init__(self, n_actions, conv_channels, n_feature_maps=12, n_hidden_units=256):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=conv_channels, out_channels=n_feature_maps, ksize=8, stride=4)
            self.conv2 = L.Convolution2D(in_channels=n_feature_maps, out_channels=n_feature_maps, ksize=4, stride=2)
            self.conv3 = L.Convolution2D(in_channels=n_feature_maps, out_channels=n_feature_maps, ksize=2, stride=1)
            self.fc1 = L.Linear(in_size=None, out_size=n_hidden_units)
            self.fc2 = L.Linear(in_size=None, out_size=n_actions)
            # self.bn1 = L.BatchNormalization(n_feature_maps)
            # self.bn2 = L.BatchNormalization(n_feature_maps)
            # self.bn3 = L.BatchNormalization(n_feature_maps)
            # self.bn4 = L.BatchNormalization(n_hidden_units)

    def __call__(self, x):
        # h = self.bn1(F.relu(self.conv1(x)))
        # h = self.bn2(F.relu(self.conv2(h)))
        # h = self.bn3(F.relu(self.conv3(h)))
        # h = self.bn4(F.relu(self.fc1(h)))
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc1(h))
        h = self.fc2(h)

        return h


class FCN(Chain):
    def __init__(self, n_actions, n_hidden_units=256):
        super(FCN, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_hidden_units)
            self.fc2 = L.Linear(n_hidden_units, n_actions)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)

        return h
