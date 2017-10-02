from .. import callback


class AdaptiveLearningRateCallback(callback.Callback):
    """
    Callback that changes the learning rate dynamically.

    lr = initial_learning_rate * (change ** (epochs_passed // num_epochs))

    Arguments:
        - initial_learning_rate : The learning rate to start with.
        - change : The value by which the learning rate is changed (change * current_learning_rate).
        - num_epochs : After how many epochs the learning rate should be updated.
    """

    def __init__(self, initial_learning_rate=0.001, change=0.3, num_epochs=1):
        super(AdaptiveLearningRateCallback, self).__init__()

        self.initial_lr = initial_learning_rate
        self.change = change
        self.num_epochs = num_epochs

    def before_train_epoch(self, epoch_index, epoch_log):
        self.current_epoch_log = epoch_log

        epochs_passed = epoch_index + 1
        new_lr = self.initial_lr * (self.change ** (epochs_passed // self.num_epochs))

        for param_group in self._trainer._optimizer:
            param_group['lr'] = new_lr
