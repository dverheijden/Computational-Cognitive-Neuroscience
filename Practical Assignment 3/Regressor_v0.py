from chainer import report
from chainer import Chain
import chainer.functions as F


class Regressor_v0(Chain):
	def __init__(self, RNN):
		super(Regressor_v0, self).__init__()
		with self.init_scope():
			self.model = RNN

	def __call__(self, x, label):
		self.y = self.model(x)
		self.loss = (self.y - label)**2
		self.accuracy = - F.log(abs(self.y - label))
		return self.loss
