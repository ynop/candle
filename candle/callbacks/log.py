import logging

from . import base


class LoggerCallback(base.Callback):
    """
    Logs training information via logging module.

    Arguments:
        batch_log_interval (int): How many batches should be accumulated for one log output.
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

        lrates = []

        for param_group in self.trainer._optimizer.param_groups:
            lrates.append(str(param_group['lr']))

        logging.info("#" * 50)
        logging.info("Epoch {} / {} (Learing-Rates {})".format(epoch_index + 1, self.trainer._num_epochs, ', '.join(lrates)))
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
