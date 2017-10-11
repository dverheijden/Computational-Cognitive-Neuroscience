from chainer import report
from chainer import Chain
import chainer.functions as F


class Regressor(Chain):
	def __init__(self, RNN):
		super(Regressor, self).__init__()
		with self.init_scope():
			self.model = RNN

	def __call__(self, x, label):
		y = self.model(x)
		return y - label
