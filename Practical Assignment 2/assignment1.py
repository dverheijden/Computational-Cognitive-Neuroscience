import utils
from chainer import optimizers
import Networks
import chainer
from Classifier import Classifier
import matplotlib.pyplot as plt
import pickle

# Import whole dataset
train, test = chainer.datasets.get_mnist()

n_output = 10
n_epochs = 20
batch_size = 32

train_iter = utils.RandomIterator(train, batch_size)
test_iter = utils.RandomIterator(test, batch_size)


def feed_data(model, optimizer, random_iter, update):
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


def trainNetwork(model, optimizer):
    """
    Trains the MLP network for n_epochs.
    One epoch contains of a training phase and testing phase. Afterwards, the results are printed to the screen
    """
    train_loss_list = []
    test_loss_list = []
    for epoch in range(n_epochs):
        train_loss, train_accuracy = feed_data(model, optimizer, train_iter, True)
        test_loss, test_accuracy = feed_data(model, optimizer, test_iter, False)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('Epoch {} \n'
              'Training: accuracy: {} \t loss: {} \n'
              'Testing: accuracy: {} \t loss: {}'.format(epoch + 1,
                                                         train_accuracy, train_loss,
                                                         test_accuracy, test_loss))

    return train_loss_list, test_loss_list


def run():
    for N in range(1, 4):
        model = Classifier(Networks.FullyConnectedNet(N, 10))
        optimizer = optimizers.SGD()
        optimizer.setup(model)

        train_loss_list, test_loss_list = trainNetwork(model, optimizer)
        plt.plot(test_loss_list, label='Test Loss')
        plt.plot(train_loss_list, label='Train Loss')
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Loss as a function of epochs, N=%s" %N)
        plt.show()


run()