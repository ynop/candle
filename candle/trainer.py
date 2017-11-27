from torch import autograd

from . import log
from . import dispatcher
from . import callbacks as cb


class Trainer(object):
    """
    The trainer is the main class to train / evaluate models.

    Arguments:
        model (torch.nn.Module) : The pytorch model to be trained or evaluated.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        targets (list) : List of targets (:py:class:`candle.Target`) to use for training/evaluation.
        num_epochs (int): Number of epochs to train.
        use_cuda (bool): Whether to use CUDA for computation.
        callbacks (list): Callbacks that should be informed about given events.
        metrics (list): Metrics which should be evaluated (name/metric)
        dispatcher (Dispatcher): The dispatcher to use (By default the :py:class:`candle.Dispatcher` is used).

    Example:
        >>> import torch
        >>> from torch.utils import data
        >>> from torch import optim
        >>>
        >>> # Create data loaders
        >>> train_loader = data.DataLoader(train_ds, batch_size=10, shuffle=True)
        >>> dev_loader = data.DataLoader(dev_ds, batch_size=10, shuffle=False)
        >>> test_loader = data.DataLoader(test_ds, batch_size=10, shuffle=False)
        >>>
        >>> # Create the model
        >>> model = torch.nn.Linear(10, 2)
        >>>
        >>> # Optimizer and loss
        >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
        >>> mse = torch.nn.MSELoss()
        >>>
        >>> # Create the trainer
        >>> trainer = Trainer(model, optimizer,
        >>>                          targets=[candle.Target('MSE', mse)],
        >>>                          num_epochs=3,
        >>>                          use_cuda=False)
        >>>
        >>> # TRAIN
        >>> train_log = trainer.train(train_loader, dev_loader)
        >>>
        >>> # EVALUATE
        >>> eval_log = trainer.evaluate(test_loader)
    """

    default_callbacks = [
        cb.LoggerCallback
    ]

    def __init__(self, model, optimizer, targets=[], num_epochs=10, use_cuda=True, callbacks=[], metrics=[], dispatcher=dispatcher.Dispatcher()):
        self._model = model
        self._optimizer = optimizer

        self._targets = targets
        self._metrics = metrics

        self._num_epochs = num_epochs
        self._use_cuda = use_cuda

        self._callback_handler = cb.CallbackHandler()
        self._callback_handler.callbacks.extend(callbacks)
        self._callback_handler.callbacks.extend(Trainer.instantiate_default_callbacks())
        self._callback_handler.set_trainer(self)

        self.dispatcher = dispatcher

    def train(self, train_loader, dev_loader=None):
        """
        Run the training.

        Arguments:
            train_loader (torch.utils.data.DataLoader): PyTorch loader that provides training data.
            dev_loader (torch.utils.data.DataLoader): PyTorch loader that provides validation data.

        Returns:
            TrainingLog: The training log.
        """
        train_log = log.TrainingLog(targets=self._targets, metrics=self._metrics)

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
            test_loader (torch.utils.data.DataLoader): PyTorch loader that provides evaluation data.

        Returns:
            IterationLog: The iteration log.
        """
        iteration_log = log.IterationLog(targets=self._targets, metrics=self._metrics)

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
            epoch_index (int): An index that identifies the epoch.
            train_loader (torch.utils.data.DataLoader): PyTorch loader that provides training data.
            dev_loader (torch.utils.data.DataLoader): PyTorch loader that provides validation data.

        Returns:
            IterationLog: The iteration log.
        """
        iteration_log = log.IterationLog(targets=self._targets, metrics=self._metrics)

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
            epoch_index (int): An index that identifies the epoch.
            batch_index (int): An index that identifies the batch.
            batch (list, torch.Tensor, ...): The data of the batch.

        Returns:
            BatchLog: The batch log.
        """
        self._callback_handler.notify_before_train_batch(epoch_index, batch_index)

        batch_log = log.BatchLog()

        batch = self.dispatcher.prepare_batch(batch, use_cuda=self._use_cuda)

        self._optimizer.zero_grad()

        output = self.dispatcher.forward(self._model, batch)

        losses = self.dispatcher.compute_losses(self._targets, output, batch)
        grads = [ls.data.new(1).fill_(1) for ls in losses]
        autograd.backward(losses, grads)

        self._optimizer.step()

        batch_log.loss = [x.data[0] for x in losses]
        batch_log.metrics = self.dispatcher.compute_metrics(self._metrics, output, batch, self._model)

        self._callback_handler.notify_after_train_batch(epoch_index, batch_index, batch_log)

        return batch_log

    def evaluate_batch(self, batch_index, batch):
        """
        Run evaluation of one batch. Returns a batch log.

        Arguments:
            batch_index (int): An index that identifies the batch.
            batch (list, torch.Tensor, ...): The data of the batch.

        Returns:
            BatchLog: The batch log.
        """
        self._callback_handler.notify_before_evaluate_batch(batch_index)

        batch = self.dispatcher.prepare_batch(batch, use_cuda=self._use_cuda)

        batch_log = log.BatchLog()

        output = self.dispatcher.forward(self._model, batch)
        losses = self.dispatcher.compute_losses(self._targets, output, batch)

        batch_log.loss = [x.data[0] for x in losses]
        batch_log.metrics = self.dispatcher.compute_metrics(self._metrics, output, batch, self._model)

        self._callback_handler.notify_after_evaluate_batch(batch_index, batch_log)

        return batch_log

    def prepare_model(self):
        """
        Prepares the model for training/evaluation.
        """
        if self._use_cuda:
            self._model.cuda()

    @classmethod
    def instantiate_default_callbacks(cls):
        """
        Create an instance of the all the default callbacks.

        Returns:
            list: List of callback instances.
        """
        def_callbacks = []

        for callback_class in cls.default_callbacks:
            def_callbacks.append(callback_class())

        return def_callbacks
