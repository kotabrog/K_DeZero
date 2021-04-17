class History:
    """Class to record learning results

    Attribute:
        loss (list): Loss per epoch
        acc (list): Accuracy per epoch
        val_loss (list): Validation loss per epoch
        val_acc (list): Validation accuracy per epoch

    Note:
        If there is no accuracy or validation, the target list will be empty.
    """
    def __init__(self):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def update(self, loss, acc=None, val_loss=None, val_acc=None):
        """Update the history."""
        self.loss.append(loss)
        if acc is not None:
            self.acc.append(acc)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_acc.append(val_acc)


class CalculateHistory:
    """Class for calculating loss and accuracy
    """
    def __init__(self, loss=None, acc=None):
        if loss is not None:
            self.loss = 0.
        else:
            self.loss = None
        if acc is not None:
            self.acc = 0.
        else:
            self.acc = None

    def reset(self):
        if self.loss is not None:
            self.loss = 0.
        if self.acc is not None:
            self.acc = 0.

    def add_hist(self, batch_size, loss=None, acc=None):
        if loss is not None:
            self.loss += float(loss.data) * batch_size
        if acc is not None:
            self.acc += float(acc.data) * batch_size

    def mean_hist(self, data_size):
        if self.loss is not None:
            self.loss /= data_size
        if self.acc is not None:
            self.acc /= data_size
