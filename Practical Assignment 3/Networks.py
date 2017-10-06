from chainer import Chain
import chainer.functions as F
import chainer.links as L


class RNN(Chain):
    def __init__(self, n_hidden):
        super(RNN, self).__init__()
        with self.init_scope():
            self.lstm = L.LSTM(None, n_hidden)  # the first LSTM layer
            self.out = L.Linear(n_hidden, 1)  # the feed-forward output layer

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x):
        l1 = self.lstm(x)
        y = self.out(l1)
        return y
