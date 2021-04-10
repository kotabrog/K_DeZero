import numpy as np
from kdezero.datasets import Dataset


class SinCurve(Dataset):
    """The data is a sine curve, and the label advances one index by one.
    The number of data is 1000.

    Attribute:
        Attribute:
        data (ndarray):
        label (ndarray):
        train (bool):
            Flag for learning.
            If train is True, data is sin and noise.
            If not, data is cos.
    Examples:
        >>> print(dataset.data[:5])
            [[-0.04955855]
            [ 0.03048039]
            [-0.01378722]
            [-0.02327317]
            [ 0.04658464]]

        >>> print(dataset.label[:5])
            [[ 0.03048039]
            [-0.01378722]
            [-0.02327317]
            [ 0.04658464]
            [ 0.02806842]]
    """
    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]
