from chainer import link
from chainer import reporter


class Regressor(link.Chain):
	compute_accuracy = True

	def __init__(self, predictor, lossfun, accfun):
		super(Regressor, self).__init__()
		self.lossfun = lossfun
		self.accfun = accfun
		self.y = None
		self.loss = None
		self.accuracy = None

		with self.init_scope():
			self.predictor = predictor

	def __call__(self, x, t, *args, **kwargs):
		self.y = None
		self.loss = None
		self.accuracy = None
		self.y = self.predictor(x)
		self.loss = self.lossfun(args, kwargs)
		reporter.report({'loss': self.loss}, self)
		if self.compute_accuracy:
			self.accuracy = self.accfun(self.y, t)
			reporter.report({'accuracy': self.accuracy}, self)
		return self.loss
