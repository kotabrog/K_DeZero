import numpy as np
import kdezero.layers as L
import kdezero.functions as F
from kdezero import utils
from kdezero import cuda
from kdezero import optimizers
from kdezero import no_grad
from kdezero import History
from kdezero.logger import CalculateHistory


class Model(L.Layer):
    """Model class
    """
    def __init__(self):
        super().__init__()
        self.done_compile = False
        self.gpu = False

    def compile(self,
                optimizer=optimizers.Adam(),
                loss=F.softmax_cross_entropy,
                acc=None):
        """Make settings for training

        Args:
            optimizer (kdezero.Optimizer, optional):
            loss (kdezero.Function, optional): Loss function.
            acc (None or kdezero.Function, optional):
                Evaluation function. 
                If None, acc will not be output to the running output.

        Returns:
            kdezero.Model: Return self.
        """
        self.loss = loss
        self.acc = acc
        self.optimizer = optimizer.setup(self)
        self.done_compile = True

    def fit_generator(self,
                      data_loader,
                      max_epoch=100,
                      gpu=False,
                      verbose=1):
        """Train with a data loader.

        Args:
            data_loader (kdezero.DataLoader):
            max_epoch (int, optional):
            gpu (bool, optional): If True, use gpu.
            verbose (int, optional):
                Determine the method of output being performed, depending on the numbers given.
                examples:
                    1. epoch: 1
                       train loss: 0.1910525824315846, accuracy: 0.9432833333333334
                       epoch: 2
                       train loss: 0.07954498826215664, accuracy: 0.97465
                       ...
        """
        if not self.done_compile:
            raise Exception("Compilation is not complete")

        self.gpu = gpu
        if cuda.gpu_enable and gpu:
            data_loader.to_gpu()
            self.to_gpu()
            if verbose > 0:
                print('set gpu')
        else:
            data_loader.to_cpu()
            self.to_cpu()

        history = History()
        calc_history = CalculateHistory(self.loss, self.acc)

        data_loader.reset()
        for epoch in range(max_epoch):
            calc_history.reset()

            for x, t in data_loader:
                y = self(x)
                loss = self.loss(y, t)
                acc = None
                if self.acc:
                    acc = self.acc(y, t)
                self.cleargrads()
                loss.backward()
                self.optimizer.update()

                calc_history.add_hist(len(t), loss, acc)

            calc_history.mean_hist(data_loader.data_size)
            history.update(calc_history.loss, calc_history.acc)

            if verbose > 0:
                print('epoch: {}'.format(epoch + 1))
                print('train loss: {}, accuracy: {}'.format(
                    calc_history.loss, calc_history.acc))

        return history

    def evaluate(self, data_loader, gpu=None):
        """Evaluate the test data set

        Args:
            data_loader (kdezero.DataLoader):
            gpu (None or bool, optional):
                If True, use gpu. If None, use the mode of train.

        Returns:
            tuple of float: return (loss, acc)
        """
        if self.acc is None:
            raise Exception("Metrics is not set.")

        if gpu is None:
            gpu = self.gpu
        if cuda.gpu_enable and gpu:
            data_loader.to_gpu()
            self.to_gpu()
        else:
            data_loader.to_cpu()
            self.to_cpu()

        calc_history = CalculateHistory(self.loss, self.acc)
        data_loader.reset()
        with no_grad():
            for x, t in data_loader:
                y = self(x)
                loss = self.loss(y, t)
                acc = self.acc(y, t)
                calc_history.add_hist(len(t), loss, acc)

        calc_history.mean_hist(data_loader.data_size)
        return calc_history.loss, calc_history.acc

    def predict(self, x, gpu=None):
        if gpu is None:
            gpu = self.gpu
        if cuda.gpu_enable and gpu:
            x = cuda.as_cupy(x)
            self.to_gpu()
        else:
            x = cuda.as_numpy(x)
            self.to_cpu()

        return self(x)

    def plot(self, *inputs, to_file='model.png'):
        """Display the calculation graph of model

        Args:
            inputs (kdezero.Variables):
            to_file (str, optional): File path (default: model.png)

        Note:
            You need to install graphviz.
        """
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class Sequential(Model):
    """A class that makes it easy to configure models sequentially.

    Attribute:
        layers (list):
            A list containing layers and functions to be processed in order.
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class VGG16(Model):
    WEIGHTS_PATH = 'https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz'

    def __init__(self, pretraind=False):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4069)
        self.fc7 = L.Linear(4069)
        self.fc8 = L.Linear(1000)

        if pretraind:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.asarray([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image
