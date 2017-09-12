from chainer import report
from chainer import Chain
import chainer.functions as F


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, data, labels):
        y = self.predictor(data)
        self.loss = F.softmax_cross_entropy(y, labels)
        self.accuracy = F.accuracy(y, labels)
        report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
