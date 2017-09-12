import utils
from chainer import optimizers
import chainer.links as L
from MLP import MLP


train, test = utils.get_mnist(100, 100)

n_hidden = 10
n_output = 10
n_epochs = 20
batch_size = 32

train_iter = utils.RandomIterator(train, batch_size)
test_iter = utils.RandomIterator(test, batch_size)

# Standard classifier that uses softmax_cross_entropy as loss function
model = L.Classifier(MLP(n_hidden, n_output))
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

        total_loss += float(model.loss.data) * len(labels)
        total_accuracy += float(model.accuracy.data) * len(labels)
    return total_loss / random_iter.idx, total_accuracy / random_iter.idx


def run():
    """
    Trains the MLP network for n_epochs.
    One epoch contains of a training phase and testing phase. Afterwards, the results are printed to the screen
    """
    for epoch in range(n_epochs):
        train_loss, train_accuracy = feed_data(train_iter, True)
        test_loss, test_accuracy = feed_data(test_iter, False)

        print('Epoch {} \n'
              'Training: accuracy: {} \t loss: {} \n'
              'Testing: accuracy: {} \t loss: {}'.format(epoch + 1,
                                                         train_accuracy, train_loss,
                                                         test_accuracy, test_loss))


run()
