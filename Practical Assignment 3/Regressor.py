from chainer import reporter
from chainer import Chain
import chainer.functions as F
from chainer.functions.evaluation import accuracy


class Regressor(Chain):
	def __init__(self, predictor, lossfun=F.softmax_cross_entropy, accfun=accuracy.accuracy):
		super(Regressor, self).__init__()
		self.lossfun = lossfun
		self.accfun = accfun
		with self.init_scope():
			self.predictor = predictor

	def __call__(self, x, label):
		self.y = self.model(x)
		self.loss = self.lossfun(self.y, label)
		self.accuracy = self.accfun(self.y, label)
		reporter.report({'accuracy': self.accuracy}, self)
