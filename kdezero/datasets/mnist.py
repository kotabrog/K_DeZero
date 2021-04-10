import numpy as np
import gzip
import matplotlib.pyplot as plt
from kdezero.datasets import Dataset
from kdezero.transforms import Compose, Flatten, ToFloat, Normalize
from kdezero.utils import get_file


class MNIST(Dataset):
    """MNISET dataset
    A class that creates a MNIST dataset by downloading data
    from the URL written in the prepare function.

    Data is stored in '.kdezero' in your home directory. (~/.kdezero)

    Attribute:
        data (ndarray):
            Shape is (n, 1, 28, 28) (n, channel, height, width).
            If train == True, n == 60000. If not, n == 10000
        label (ndarray): shape is (n) and element is int.
        train (bool): Flag for learning.
        transform (Callable or None, optional):
            Object to convert data.
            By default, flatten and to float and normalize (mean is 0, std is 255)
        target_transform (Callable, optional):
            Object to convert label. (default: None)
    """
    def __init__(self,
                 train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        """Download the data and prepare the MNIST dataset.
        """
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}
        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        """Display data randomly

        Specify the width and height, then display the data randomly for that number.

        Args:
            row (int): Number of images side by side
            col (int): Number of images arranged vertically

        Note:
            The image shall be flatten processed.
        """
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H: (r + 1) * H, c * W: (c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)
                ].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    def one_show(self, index):
        """Display image by specifying index

        Args:
            index (int): Index of data.
        """
        H, W = 28, 28
        img = self.data[index].reshape(H, W)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
