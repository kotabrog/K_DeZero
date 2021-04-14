import numpy as np
import kdezero.layers as L
import kdezero.functions as F
from kdezero import utils
from kdezero import cuda
from kdezero import optimizers


class Model(L.Layer):
    """Model class
    """
    def __init__(self):
        super().__init__()
        self.done_compile = False


    def compile(self,
              optimizer=optimizers.Adam(),
              loss=F.softmax_cross_entropy,
              max_epoch=100,
              acc=None,
              gpu=False,
              verbose=1):
        """Make settings for training

        Args:
            optimizer (kdezero.Optimizer, optional):
            loss (kdezero.Function): Loss function.
            max_epoch (int):
            acc (None or kdezero.Function):
                Evaluation function. 
                If None, acc will not be output to the running output.
            gpu (bool): If True, use gpu.
            verbose (int):
                Determine the method of output being performed, depending on the numbers given.
                examples:
                    1. epoch: 1
                       train loss: 0.1910525824315846, accuracy: 0.9432833333333334
                       epoch: 2
                       train loss: 0.07954498826215664, accuracy: 0.97465
                       ...

        Returns:
            kdezero.Model: Return self.
        """
        self.loss = loss
        self.max_epoch = max_epoch
        self.acc = acc
        self.gpu = gpu
        self.verbose = verbose
        self.optimizer = optimizer.setup(self)
        self.done_compile = True

    def fit_generator(self, data_loader):
        """Train with a data loader.

        Args:
            data_loader (kdezero.DataLoader):
        """
        if not self.done_compile:
            raise Exception("Compilation is not complete")

        if cuda.gpu_enable and self.gpu:
            data_loader.to_gpu()
            self.to_gpu()
            if self.verbose > 0:
                print('set gpu')
        
        for epoch in range(self.max_epoch):
            sum_loss, sum_acc = 0, 0

            for x, t in data_loader:
                y = self(x)
                loss = self.loss(y, t)
                if self.acc:
                    acc = self.acc(y, t)
                self.cleargrads()
                loss.backward()
                self.optimizer.update()

                sum_loss += float(loss.data) * len(t)
                if self.acc:
                    sum_acc += float(acc.data) * len(t)

            if self.verbose > 0:
                print('epoch: {}'.format(epoch + 1))
                print('train loss: {}, accuracy: {}'.format(
                    sum_loss / data_loader.data_size, sum_acc / data_loader.data_size))

        return self

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
            setattr(self, 'l' + str(i), layers)
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
