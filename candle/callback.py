import logging


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


class LoggerCallback(Callback):
    """
    Logs training information via logging module.
    """

    def __init__(self, batch_log_interval=30):
        super(LoggerCallback, self).__init__()

        self.batch_log_interval = batch_log_interval

        self.is_training = False

    def before_train(self, training_log):
        super(LoggerCallback, self).before_train(training_log)

        logging.info("#" * 50)
        logging.info("# Start Training")
        logging.info("#" * 50)

        self.is_training = True

    def after_train(self, training_log):
        super(LoggerCallback, self).after_train(training_log)

        logging.info("Finished Training")

        self.is_training = False

    def before_train_epoch(self, epoch_index, epoch_log):
        super(LoggerCallback, self).before_train_epoch(epoch_index, epoch_log)

        logging.info("#" * 50)
        logging.info("Epoch {} / {}".format(epoch_index + 1, self._trainer._num_epochs))
        logging.info("#" * 50)

    def after_train_epoch(self, epoch_index, epoch_log):
        super(LoggerCallback, self).after_train_epoch(epoch_index, epoch_log)

        logging.info("-" * 50)
        logging.info("Epoch stats:")
        logging.info("  TRAINED BATCHES: {}".format(len(epoch_log.batches)))

        for line in epoch_log.stats().split('\n'):
            logging.info("  {}".format(line))

        logging.info("-" * 50)

    def before_train_batch(self, epoch_index, batch_index):
        super(LoggerCallback, self).before_train_batch(epoch_index, batch_index)
        pass

    def after_train_batch(self, epoch_index, batch_index, batch_log):
        super(LoggerCallback, self).after_train_batch(epoch_index, batch_index, batch_log)

        if batch_index % self.batch_log_interval == self.batch_log_interval - 1:
            mean_losses = self.current_epoch_log.mean_loss_with_names(recent=self.batch_log_interval)

            loss_info = ' / '.join(['{} = {}'.format(x[0], x[1]) for x in mean_losses.items()])
            logging.info('[Trained batches: {}] training loss: {}'.format(batch_index + 1, loss_info))

    def before_evaluate(self, iteration_log):
        super(LoggerCallback, self).before_evaluate(iteration_log)

        if not self.is_training:
            logging.info("#" * 50)
            logging.info("# Start Evaluation")
            logging.info("#" * 50)

    def after_evaluate(self, iteration_log):
        super(LoggerCallback, self).after_evaluate(iteration_log)

        if not self.is_training:
            for line in iteration_log.stats().split('\n'):
                logging.info("  {}".format(line))

            logging.info("-" * 50)
