

class Callback(object):
    """
    Class that gets informed and can react on given events during the training.
    """

    def __init__(self):
        self._trainer = None

        self.current_train_log = None
        self.current_epoch_log = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def before_train(self, training_log):
        self.current_train_log = training_log

    def after_train(self, training_log):
        self.current_train_log = None

    def before_train_epoch(self, epoch_index, epoch_log):
        self.current_epoch_log = epoch_log

    def after_train_epoch(self, epoch_index, epoch_log):
        self.current_epoch_log = None

    def before_train_batch(self, epoch_index, batch_index):
        pass

    def after_train_batch(self, epoch_index, batch_index, batch_log):
        pass

    def before_evaluate(self, iteration_log):
        pass

    def after_evaluate(self, iteration_log):
        pass

    def before_evaluate_batch(self, batch_index):
        pass

    def after_evaluate_batch(self, batch_index, batch_log):
        pass


class CallbackHandler(object):
    """
    Class that stores a list of callbacks and handles notification distribution.
    """

    def __init__(self):
        self.callbacks = []

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def notify_before_train(self, training_log):
        for callback in self.callbacks:
            callback.before_train(training_log)

    def notify_after_train(self, training_log):
        for callback in self.callbacks:
            callback.after_train(training_log)

    def notify_before_train_epoch(self, epoch_index, epoch_log):
        for callback in self.callbacks:
            callback.before_train_epoch(epoch_index, epoch_log)

    def notify_after_train_epoch(self, epoch_index, epoch_log):
        for callback in self.callbacks:
            callback.after_train_epoch(epoch_index, epoch_log)

    def notify_before_train_batch(self, epoch_index, batch_index):
        for callback in self.callbacks:
            callback.before_train_batch(epoch_index, batch_index)

    def notify_after_train_batch(self, epoch_index, batch_index, batch_log):
        for callback in self.callbacks:
            callback.after_train_batch(epoch_index, batch_index, batch_log)

    def notify_before_evaluate(self, iteration_log):
        for callback in self.callbacks:
            callback.before_evaluate(iteration_log)

    def notify_after_evaluate(self, iteration_log):
        for callback in self.callbacks:
            callback.after_evaluate(iteration_log)

    def notify_before_evaluate_batch(self, batch_index):
        for callback in self.callbacks:
            callback.before_evaluate_batch(batch_index)

    def notify_after_evaluate_batch(self, batch_index, batch_log):
        for callback in self.callbacks:
            callback.after_evaluate_batch(batch_index, batch_log)

