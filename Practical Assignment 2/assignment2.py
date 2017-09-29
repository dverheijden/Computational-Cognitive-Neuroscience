import utils
from chainer import optimizers
import Networks
import chainer
from chainer.links import Classifier
import matplotlib.pyplot as plt
import pickle

# Import whole dataset
train, test = chainer.datasets.get_mnist(ndim=3)

n_hidden = 10
n_output = 10
n_epochs = 20
batch_size = 32

train_iter = utils.RandomIterator(train, batch_size)
test_iter = utils.RandomIterator(test, batch_size)


def get_model(model_name):
    try:
        pickle_in = open("{}_model.pickle".format(model_name), 'rb')
        model = pickle.load(pickle_in)

        pickle_in = open("{}_optimizer.pickle".format(model_name), 'rb')
        optimizer = pickle.load(pickle_in)

        pickle_in = open("{}_results.pickle".format(model_name), 'rb')
        results = pickle.load(pickle_in)
    except FileNotFoundError:
        # Standard classifier that uses softmax_cross_entropy as loss function
        model = Classifier(Networks.Convolutional())
        optimizer = optimizers.SGD()
        optimizer.setup(model)
        print("Model not found! Starting training ...")
        results = train_network(model, optimizer)
        with open('{}_model.pickle'.format(model_name), 'wb') as f:
            pickle.dump(model, f)
        with open('{}_optimizer.pickle'.format(model_name), 'wb') as f:
            pickle.dump(optimizer, f)
        with open('{}_results.pickle'.format(model_name), 'wb') as f:
            pickle.dump(results, f)

    return model, optimizer, results


def feed_data(model, optimizer, random_iter, update):
    """
    Feeds the network data
    :param random_iter: Iterator that holds the data
    :param update: Boolean whether to update the model parameters
    :return: loss and accuracy
    """
    with chainer.using_config('train', update):
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


def train_network(model, optimizer):
    """
    Trains the network for n_epochs.
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
    return [train_loss_list, test_loss_list]


def test_network(model, optimizer):
    """
    Tests the conv network
    """

    train_loss, train_accuracy = feed_data(model, optimizer, train_iter, False)
    test_loss, test_accuracy = feed_data(model, optimizer, test_iter, False)

    print('Training Data: accuracy: {} \t loss: {} \n'
          'Testing Data: accuracy: {} \t loss: {}'.format(train_accuracy, train_loss,
                                                          test_accuracy, test_loss))


# MAIN PROGRAM START
model, optimizer, results = get_model("convnet_v0")
test_network(model, optimizer)

plt.plot(results[1], label='Test Loss')
plt.plot(results[0], label='Train Loss')
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Loss as a function of epochs")
plt.show()
