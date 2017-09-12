import collections

import torch
from torch import autograd


class Dispatcher(object):
    """
    The Dispatcher is used to customize the behaviour/handling of the data processing.

    The default dispatcher expects the following behaviour:
        - The input tensor is available via batch[0]
        - The target tensor is available via batch[1]

    """

    def __init__(self, prepare_batch_func=None, forward_func=None, compute_losses_func=None, compute_metrics_func=None):
        self.prepare_batch_func = prepare_batch_func
        self.forward_func = forward_func
        self.compute_losses_func = compute_losses_func
        self.compute_metrics_func = compute_metrics_func

    def prepare_batch(self, batch, use_cuda=True):
        """
        The prepare batch function is called when a batch is grabbed from the dataloader before its passed to the forward function.

        It can/should be used:
            * to wrap tensors in Variables
            * to move tensors to GPU

        Arguments:
            - batch: The batch as grabbed from the dataloader.
            - use_cuda: If Variables/Tensors should be moved to GPU

        """
        if self.prepare_batch_func is not None:
            return self.prepare_batch_func(batch, use_cuda=use_cuda)
        else:
            return self.do_prepare_batch(batch, use_cuda=use_cuda)

    def forward(self, model, batch):
        """
        Run a forward pass for one batch and return the output.

        Arguments:
            - model: The model from the trainer
            - batch: The batch processed with the prepare_batch function

        """
        if self.forward_func is not None:
            return self.forward_func(model, batch)
        else:
            return self.do_forward(model, batch)

    def compute_losses(self, losses, output, batch):
        """
        Compute the losses for the given output and batch data.

        Arguments:
            - losses : List of tuples (loss-name, loss-function)
            - output : The output returned from the forward function
            - batch : The batch processed with the prepare_batch function
        """
        if self.compute_losses_func is not None:
            return self.compute_losses_func(losses, output, batch)
        else:
            return self.do_compute_losses(losses, output, batch)

    def compute_metrics(self, metrics, output, batch):
        """
        Compute the metrics for the given output and batch data.

        Arguments:
            - metrics : List of tuples (metric-name, metric-instance)
            - output : The output returned from the forward function
            - batch : The batch processed with the prepare_batch function
        """

        if self.compute_metrics_func is not None:
            return self.compute_metrics_func(metrics, output, batch)
        else:
            return self.do_compute_metrics(metrics, output, batch)

    def do_prepare_batch(self, batch, use_cuda=True):
        data = batch

        if isinstance(data, collections.Sequence):
            data = [self.do_prepare_batch(x, use_cuda=use_cuda) for x in data]
        elif torch.is_tensor(data):
            if use_cuda:
                data = data.cuda()

            data = autograd.Variable(data)

        return data

    def do_forward(self, model, batch):
        return model.forward(batch[0])

    def do_compute_losses(self, losses, output, batch):
        target = batch[1]

        return [loss_func(output, target) for __, loss_func in losses]

    def do_compute_metrics(self, metrics, output, batch):
        output = output.data
        target = batch[1].data

        return [metric.compute(output, target) for __, metric in metrics]
