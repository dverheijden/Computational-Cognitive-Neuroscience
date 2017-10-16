import numpy as np

from chainer.datasets import TupleDataset


class StreamingIterator(object):
	"""
	Generates random subsets of data
	"""

	def __init__(self, data, batch_size=20):
		"""
		Args:
			data (TupleDataset):
			batch_size (int):

		Returns:
			list of batches consisting of (input, output) pairs
		"""

		self.data = data

		self.batch_size = batch_size
		self.n_batches = len(self.data) // batch_size

	def __iter__(self):
		self.idx = -1
		self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]
		self._data_ = [self.data[x * self.batch_size:(x + 1) * self.batch_size] for x in range(self.n_batches)]
		np.random.shuffle(self._data_)

		return self

	def __next__(self):
		self.idx += 1

		if self.idx == self.n_batches:
			raise StopIteration

		i = self.idx

		return self._data_[i]

	def __len__(self):
		return self.n_batches
