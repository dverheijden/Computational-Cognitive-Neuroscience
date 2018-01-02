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


class ProgNet_Toy(Chain):
    def __init__(self, n_actions, scale=1):
        super(ProgNet_Toy, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=5, ksize=4, stride=2)
            self.fc1 = L.Linear(in_size=None, out_size=256)
            self.fc2 = L.Linear(in_size=None, out_size=n_actions)

    def __call__(self, x):
        conv_output = F.relu(self.conv(x))
        max_pooling_output = F.max_pooling_2d(conv_output, ksize=2)
        lin_output = self.lin(max_pooling_output)

        return lin_output


class ProgNet(Chain):
    def __init__(self, n_actions, n_feature_maps=12, n_hidden_units=256):
        super(ProgNet, self).__init__()
        with self.init_scope():
            self.conv11 = L.Convolution2D(in_channels=3, out_channels=n_feature_maps, ksize=8, stride=4)
            self.conv12 = L.Convolution2D(in_channels=n_feature_maps, out_channels=n_feature_maps, ksize=4, stride=2)
            self.conv13 = L.Convolution2D(in_channels=n_feature_maps, out_channels=n_feature_maps, ksize=2, stride=1)
            self.fc11 = L.Linear(in_size=None, out_size=n_hidden_units)
            self.fc12 = L.Linear(in_size=None, out_size=n_actions)
            self.bn11 = L.BatchNormalization(n_feature_maps)
            self.bn12 = L.BatchNormalization(n_feature_maps)
            self.bn13 = L.BatchNormalization(n_feature_maps)
            self.bn14 = L.BatchNormalization(n_hidden_units)

        # self.conv21 = L.ConvolutionND(3, in_channels=None, out_channels=n_feature_maps, ksize=8, stride=4)
        # self.conv22 = L.ConvolutionND(3, in_channels=None, out_channels=n_feature_maps, ksize=4, stride=2)
        # self.conv23 = L.ConvolutionND(3, in_channels=None, out_channels=n_feature_maps, ksize=(3, 4), stride=1)
        # self.fc21 = L.Linear(in_size=None, out_size=n_hidden_units)
        # self.fc22 = L.Linear(in_size=None, out_size=n_actions)

        # self.conv31 = L.ConvolutionND(3, in_channels=None, out_channels=n_feature_maps, ksize=8, stride=4)
        # self.conv32 = L.ConvolutionND(3, in_channels=None, out_channels=n_feature_maps, ksize=4, stride=2)
        # self.conv33 = L.ConvolutionND(3, in_channels=None, out_channels=n_feature_maps, ksize=(3, 4), stride=1)
        # self.fc31 = L.Linear(in_size=None, out_size=n_hidden_units)
        # self.fc32 = L.Linear(in_size=None, out_size=n_actions)

    def __call__(self, x, task):
        x = x.astype('f')
        if task == 1:
            # self.set_active_task(task)
            self.output11 = self.bn11(F.relu(self.conv11(x)))
            self.output12 = self.bn12(F.relu(self.conv12(self.output11)))
            self.output13 = self.bn13(F.relu(self.conv13(self.output12)))
            self.output14 = self.bn14(F.relu(self.fc11(self.output13)))
            self.output = self.fc12(self.output14)

        if task == 2:
            self.set_active_task(task)
            self.output21 = F.relu(self.conv21(x))

        return self.output

    def disable_column_update(self, column):
        if column == 1:
            self.conv11.disable_update()
            self.conv12.disable_update()
            self.conv13.disable_update()
            self.fc11.disable_update()
            self.fc12.disable_update()

        if column == 2:
            self.conv21.disable_update()
            self.conv22.disable_update()
            self.conv23.disable_update()
            self.fc21.disable_update()
            self.fc22.disable_update()

        if column == 3:
            self.conv31.disable_update()
            self.conv32.disable_update()
            self.conv33.disable_update()
            self.fc31.disable_update()
            self.fc32.disable_update()

    def enable_column_update(self, column):
        if column == 1:
            self.conv11.enable_update()
            self.conv12.enable_update()
            self.conv13.enable_update()
            self.fc11.enable_update()
            self.fc12.enable_update()

        if column == 2:
            self.conv21.enable_update()
            self.conv22.enable_update()
            self.conv23.enable_update()
            self.fc21.enable_update()
            self.fc22.enable_update()

        if column == 3:
            self.conv31.enable_update()
            self.conv32.enable_update()
            self.conv33.enable_update()
            self.fc31.enable_update()
            self.fc32.enable_update()

    def set_active_task(self, task):
        if task == 1:
            self.enable_column_update(1)
            self.disable_column_update(2)
            self.disable_column_update(3)

        if task == 2:
            self.enable_column_update(2)
            self.disable_column_update(1)
            self.disable_column_update(3)

        if task == 3:
            self.enable_column_update(3)
            self.disable_column_update(1)
            self.disable_column_update(1)
