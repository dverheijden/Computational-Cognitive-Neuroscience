import utils
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from MLP import MLP


train, test = utils.get_mnist(100, 100)

n_hidden = 10
n_output = 10
n_epochs = 20
batch_size = 32

train_iter = utils.RandomIterator(train, batch_size)
test_iter = utils.RandomIterator(test, batch_size)

model = L.Classifier(MLP(n_hidden, n_output))
optimizer = optimizers.SGD()
optimizer.setup(model)


def run():

    total_loss = 0
    total_accuracy = 0

    for epoch in range(n_epochs):
        for data in train_iter:
            x = data[0]
            labels = data[1]
            optimizer.update(model, x, labels)

            total_loss += float(model.loss.data) * len(labels)
            total_accuracy += float(model.accuracy.data) * len(labels)

        print('Train {}: accuracy: {} \t loss: {}'.format(epoch + 1, total_accuracy / train_iter.idx, total_loss / train_iter.idx))

        total_loss = 0
        total_accuracy = 0

        for data in test_iter:
            x = data[0]
            labels = data[1]

            loss = model(x, labels)
            total_loss += float(loss.data) * len(labels)
            total_accuracy += float(model.accuracy.data) * len(labels)

        print('Test {}: accuracy: {} \t loss: {}'.format(epoch + 1, total_accuracy / test_iter.idx, total_loss / test_iter.idx))

run()
