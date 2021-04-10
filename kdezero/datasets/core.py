import numpy as np


class Dataset:
    """Class that stores data.
    This class can also preprocess the data.

    Attribute:
        train (bool, optional): Flag for learning. (default: True)
        transform (Callable or None, optional):
            Object to convert data. (default: None)
        target_transform (Callable, optional):
            Object to convert label. (default: None)
        data (ndarray or list):
            Set data in the inherited class.
        label (ndarray or list or None):
            Set label in the inherited class.
    """
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        """returns data and label

        Returns:
            tuple of data and label.
            If there is no label, tuple of data and None.
        """
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass
