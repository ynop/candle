from . import base


class AdaptiveLearningRateCallback(base.Callback):
    """
    Callback that changes the learning rate dynamically.

    lr = initial_learning_rate * (change ** (epochs_passed // num_epochs))

    Arguments:
        initial_learning_rate (float): The learning rate to start with.
        change (float): The value by which the learning rate is changed (change * current_learning_rate).
        num_epochs (int): After how many epochs the learning rate should be updated.
    """

    def __init__(self, initial_learning_rate=0.001, change=0.3, num_epochs=1):
        super(AdaptiveLearningRateCallback, self).__init__()

        self.initial_lr = initial_learning_rate
        self.change = change
        self.num_epochs = num_epochs

    def before_train_epoch(self, epoch_index, epoch_log):
        self.current_epoch_log = epoch_log

        new_lr = self.initial_lr * (self.change ** (epoch_index // self.num_epochs))

        for param_group in self.trainer._optimizer.param_groups:
            param_group['lr'] = new_lr
