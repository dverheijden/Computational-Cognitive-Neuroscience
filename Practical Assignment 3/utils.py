import numpy as np

import chainer
from chainer.datasets import TupleDataset

# Custom iterator
class StreamingIterator(object):
    """
    Generates random subsets of data
    """

    def __init__(self, data, batch_size=20):
        """
        [
        x1 => 0
        x2 => 2
        ]
        [
        x3 => 17
        x4 => 5
        ]
        ...

        xn => 12


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
        self._data_ = [self.data[x*self.batch_size:(x+1)*self.batch_size] for x in range(self.n_batches)]
        np.random.shuffle(self._data_)

        return self

    def __next__(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx 

        return self._data_[i]
        # handles unlabeled and labeled data
        if isinstance(self.data, np.ndarray):
            return self._data_[[self._order[i:(i + self.batch_size)]]]
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]])

# for i in StreamingIterator([[1,2],[3,4],[5,6],[1,2],[3,4],[5,6],[7,8],[9,10]], 2):
#     print(i)