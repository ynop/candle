class Callback(object):
    """
    Class that gets informed and can react on given events during the training.

    Attributes:
        trainer (Trainer): The trainer which is in charge of this callback.
        current_train_log (TrainingLog): Holds the latest training-log.
        current_epoch_log (IterationLog): Holds the latest epoch-log.
    """

    def __init__(self):
        self.trainer = None

        self.current_train_log = None
        self.current_epoch_log = None

    def before_train(self, training_log):
        """
        Gets called before the training starts.

        Args:
            training_log (TrainingLog): The training-log of the current training.
        """
        self.current_train_log = training_log

    def after_train(self, training_log):
        """
        Gets called after the training is finished.

        Args:
            training_log (TrainingLog): The training-log of the current training.
        """
        self.current_train_log = None

    def before_train_epoch(self, epoch_index, epoch_log):
        """
        Gets called before the training of a new epoch is started.

        Args:
            epoch_index (int): The index of the new epoch.
            epoch_log (IterationLog): The log of the new epoch.
        """
        self.current_epoch_log = epoch_log

    def after_train_epoch(self, epoch_index, epoch_log):
        """
        Gets called after the training of an epoch has finished.

        Args:
            epoch_index (int): The index of the new epoch.
            epoch_log (IterationLog): The log of the new epoch.
        """
        self.current_epoch_log = None

    def before_train_batch(self, epoch_index, batch_index):
        """
        Gets called before the training of a new batch is started.

        Args:
            epoch_index (int): The index of the current epoch.
            batch_index (int): The index of the new batch.
        """
        pass

    def after_train_batch(self, epoch_index, batch_index, batch_log):
        """
        Gets called after the training of a batch has finished.

        Args:
            epoch_index (int): The index of the current epoch.
            batch_index (int): The index of the finished batch.
            batch_log (BatchLog): The log of the finished batch.
        """
        pass

    def before_evaluate(self, iteration_log):
        """
        Gets called before an evaluation iteration starts.

        Args:
            iteration_log (IterationLog): The log of the new evaluation.
        """
        pass

    def after_evaluate(self, iteration_log):
        """
        Gets called after an evaluation iteration has finished.

        Args:
            iteration_log (IterationLog): The log of the finished evaluation.
        """
        pass

    def before_evaluate_batch(self, batch_index):
        """
        Gets called before the evaluation of a new batch starts.

        Args:
            batch_index (int): The index of the new batch.
        """
        pass

    def after_evaluate_batch(self, batch_index, batch_log):
        """
        Gets called after the evaluation of a batch has finished.

        Args:
            batch_index (int): The index of the finished batch.
        """
        pass


class CallbackHandler(object):
    """
    Class that stores a list of callbacks and handles notification distribution.
    """

    def __init__(self):
        self.callbacks = []

    def set_trainer(self, trainer):
        for callback in self.callbacks:
            callback.trainer = trainer

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
