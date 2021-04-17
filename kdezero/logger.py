class History:
    def __init__(self):
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    # def calc_sum_hist(self, batch_size, loss, acc=None, val_loss=None, val_acc=None):
    #     self.sum_hist['loss'] += loss * batch_size
    #     if acc is not None:
    #         self.sum_hist['acc'] += acc * batch_size
    #     if val_loss is not None:
    #         self.sum_hist['val_loss'] += val_loss * batch_size
    #     if val_acc is not None:
    #         self.sum_hist['val_acc'] += val_acc * batch_size

    def update(self, loss, acc=None, val_loss=None, val_acc=None):
        self.loss.append(loss)
        if acc is not None:
            self.acc.append(acc)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_acc is not None:
            self.val_acc.append(val_acc)
