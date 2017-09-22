import utils
from chainer import optimizers
import Networks
import chainer
from Classifier import Classifier
import matplotlib.pyplot as plt

# Import whole dataset
train, test = chainer.datasets.get_mnist()

n_output = 10
n_epochs = 20
batch_size = 32

train_iter = utils.RandomIterator(train, batch_size)
test_iter = utils.RandomIterator(test, batch_size)

train_loss_list = []
test_loss_list = []


# Standard classifier that uses softmax_cross_entropy as loss function
model = Classifier(Networks.FullyConnectedNet(2, 10))
optimizer = optimizers.SGD()
optimizer.setup(model)


def feed_data(random_iter, update):
    """
    Feeds the network data
    :param random_iter: Iterator that holds the data
    :param update: Boolean whether to update the model parameters
    :return: loss and accuracy
    """
    total_loss = 0
    total_accuracy = 0

    for data in random_iter:
        x = data[0]
        labels = data[1]
        if update:
            optimizer.update(model, x, labels)
        else:
            model(x, labels)
        total_loss += float(model.loss.data)
        total_accuracy += float(model.accuracy.data)
    return total_loss / random_iter.idx, total_accuracy / random_iter.idx


def run():
    """
    Trains the MLP network for n_epochs.
    One epoch contains of a training phase and testing phase. Afterwards, the results are printed to the screen
    """
    for epoch in range(n_epochs):
        train_loss, train_accuracy = feed_data(train_iter, True)
        test_loss, test_accuracy = feed_data(test_iter, False)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('Epoch {} \n'
              'Training: accuracy: {} \t loss: {} \n'
              'Testing: accuracy: {} \t loss: {}'.format(epoch + 1,
                                                         train_accuracy, train_loss,
                                                         test_accuracy, test_loss))


run()
plt.plot(test_loss_list, label='Test Loss')
plt.plot(train_loss_list, label='Train Loss')
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Loss as a function of epochs")
plt.show()
