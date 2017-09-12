from . import callback
from . import log
from . import dispatcher


class Trainer(object):
    """
    Handles the training/evaluation of a given model.

    Arguments:
        - model : The pytorch model to be trained or evaluated.
        - optimizer : The optimizer to be used for training.
        - loss_funcs : The loss functions to be applied (name/loss_func).
        - num_epochs : Number of epochs to train.
        - use_cuda : Whether to use CUDA for computation.
        - callbacks : Callbacks that should be informed about given events.
        - metrics : Metrics which should be evaluated (name/metric)

    """

    default_callbacks = [
        callback.LoggerCallback
    ]

    def __init__(self, model, optimizer, loss_funcs={}, num_epochs=10, use_cuda=True, callbacks=[], metrics={}, dispatcher=dispatcher.Dispatcher()):
        self._model = model
        self._optimizer = optimizer

        self._loss_funcs = list(loss_funcs.items())
        self._loss_names = [x[0] for x in self._loss_funcs]

        self._metrics = list(metrics.items())

        self._num_epochs = num_epochs
        self._use_cuda = use_cuda

        self._callback_handler = callback.CallbackHandler()
        self._callback_handler.callbacks.extend(callbacks)
        self._callback_handler.callbacks.extend(Trainer.instantiate_default_callbacks())
        self._callback_handler.set_trainer(self)

        self.dispatcher = dispatcher

    def train(self, train_loader, dev_loader=None):
        """
        Run the training. Returns a training log.

        - Arguments:
            - train_loader : PyTorch loader that provides training data.
            - dev_loader : PyTorch loader that provides validation data.

        """
        train_log = log.TrainingLog(losses=self._loss_names, metrics=self._metrics)

        self._callback_handler.notify_before_train(train_log)

        self.prepare_model()

        for epoch_index in range(self._num_epochs):
            iteration_log = self.train_epoch(epoch_index, train_loader, dev_loader)
            train_log.append_epoch_log(iteration_log)

        self._callback_handler.notify_after_train(train_log)

        return train_log

    def evaluate(self, test_loader):
        """
        Run evaluation. Returns a iteration log.

        Arguments:
            - test_loader : PyTorch loader that provides evaluation data.
        """
        iteration_log = log.IterationLog(losses=self._loss_names, metrics=self._metrics)

        self._callback_handler.notify_before_evaluate(iteration_log)

        self.prepare_model()

        self._model.eval()

        for batch_index, batch in enumerate(test_loader):
            batch_log = self.evaluate_batch(batch_index, batch)
            iteration_log.append_batch_log(batch_log)

        self._callback_handler.notify_after_evaluate(iteration_log)

        return iteration_log

    def train_epoch(self, epoch_index, train_loader, dev_loader=None):
        """
        Run one epoch of training. Returns a iteration log.

        Arguments:
            - epoch_index : An index that identifies the epoch.
            - train_loader : PyTorch loader that provides training data.
            - dev_loader : PyTorch loader that provides validation data.
        """
        iteration_log = log.IterationLog(losses=self._loss_names, metrics=self._metrics)

        self._model.train()
        self._callback_handler.notify_before_train_epoch(epoch_index, iteration_log)

        for batch_index, batch in enumerate(train_loader):
            batch_log = self.train_batch(epoch_index, batch_index, batch)
            iteration_log.append_batch_log(batch_log)

        dev_log = self.evaluate(dev_loader)
        iteration_log.dev_log = dev_log

        self._callback_handler.notify_after_train_epoch(epoch_index, iteration_log)

        return iteration_log

    def train_batch(self, epoch_index, batch_index, batch):
        """
        Run training of one batch. Returns a batch log.

        Arguments:
            - epoch_index : An index that identifies the epoch.
            - batch_index : An index that identifies the batch.
            - batch : The data of the batch.
        """
        self._callback_handler.notify_before_train_batch(epoch_index, batch_index)

        batch_log = log.BatchLog()

        batch = self.dispatcher.prepare_batch(batch, use_cuda=self._use_cuda)

        self._optimizer.zero_grad()

        output = self.dispatcher.forward(self._model, batch)

        losses = self.dispatcher.compute_losses(self._loss_funcs, output, batch)
        loss = self.cumulate_losses(losses)
        loss.backward()

        self._optimizer.step()

        batch_log.loss = [x.data[0] for x in losses]
        batch_log.metrics = self.dispatcher.compute_metrics(self._metrics, output, batch)

        self._callback_handler.notify_after_train_batch(epoch_index, batch_index, batch_log)

        return batch_log

    def evaluate_batch(self, batch_index, batch):
        """
        Run evaluation of one batch. Returns a batch log.

        Arguments:
            - batch_index : An index that identifies the batch.
            - batch : The data of the batch.
        """
        self._callback_handler.notify_before_evaluate_batch(batch_index)

        batch = self.dispatcher.prepare_batch(batch, use_cuda=self._use_cuda)

        batch_log = log.BatchLog()

        output = self.dispatcher.forward(self._model, batch)
        losses = self.dispatcher.compute_losses(self._loss_funcs, output, batch)

        batch_log.loss = [x.data[0] for x in losses]
        batch_log.metrics = self.dispatcher.compute_metrics(self._metrics, output, batch)

        self._callback_handler.notify_after_evaluate_batch(batch_index, batch_log)

        return batch_log

    def cumulate_losses(self, losses):
        """
        Sum up all losses, so one loss is given for backprop.
        """
        loss = losses[0]

        for i in range(1, len(losses)):
            loss += losses[i]

        return loss

    def prepare_model(self):
        """
        Prepares the model for training/evaluation.
        """
        if self._use_cuda:
            self._model.cuda()

    @classmethod
    def instantiate_default_callbacks(cls):
        def_callbacks = []

        for callback_class in cls.default_callbacks:
            def_callbacks.append(callback_class())

        return def_callbacks
