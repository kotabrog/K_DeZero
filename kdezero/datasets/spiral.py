import numpy as np
from kdezero.datasets import Dataset


def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    """Spiral data for classification problems.
    The number of data is 100.
    The number of class is 3.
    And the number of dimensions is 2.

    Attribute:
        Attribute:
        data (ndarray): shape is (300, 2)
        label (ndarray): shape is (300,)
        train (bool):
            Flag for learning.
            The internal random number is changed depending on whether train is True or not.
    """
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
