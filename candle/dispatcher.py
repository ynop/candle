import collections

import torch
from torch import autograd


class Dispatcher(object):
    """
    The Dispatcher is used to customize the behaviour/handling of the data processing.

    The default dispatcher performs the following tasks:

    * In ``prepare_batch`` it create autograd.Variable objects from the batches returned by the dataloader. Furthermore it moves these to the gpu if cuda enabled.
    * In ``forward`` the input variable is fed into the forward function of the model. The input variable is expected to be retrieved with ``batch[0]``.
    * In ``compute_losses`` it computes the loss for every target.
    * In ``compute_metrics`` it computes the result of all the metrics.

    Args:
        prepare_batch_func (func): The function used to prepare a batch. (If None the do_prepare_batch method is used)
        forward_func (func): The function used to do a forward pass. (If None the do_forwad method is used)
        compute_losses_func (func): The function used for computing losses. (If None the do_compute_losses method is used)
        compute_metrics_func (func): The function used for computing metrics. (If None the do_compute_metrics method is used)
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
            batch (list, torch.Tensor, ...): The batch as grabbed from the dataloader.
            use_cuda (bool): If Variables/Tensors should be moved to GPU

        """
        if self.prepare_batch_func is not None:
            return self.prepare_batch_func(batch, use_cuda=use_cuda)
        else:
            return self.do_prepare_batch(batch, use_cuda=use_cuda)

    def forward(self, model, batch):
        """
        Run a forward pass for one batch and return the output.

        Arguments:
            model (torch.nn.Module): The model from the trainer
            batch (torch.autograd.Variable): The batch processed with the prepare_batch function

        """
        if self.forward_func is not None:
            return self.forward_func(model, batch)
        else:
            return self.do_forward(model, batch)

    def compute_losses(self, targets, output, batch):
        """
        Compute the losses for the given output and batch data.

        Arguments:
            targets (list): List of targets to compute
            output (torch.autograd.Variable): The output returned from the forward function
            batch (torch.autograd.Variable): The batch processed with the prepare_batch function
        """
        if self.compute_losses_func is not None:
            return self.compute_losses_func(targets, output, batch)
        else:
            return self.do_compute_losses(targets, output, batch)

    def compute_metrics(self, metrics, output, batch, model):
        """
        Compute the metrics for the given output and batch data.

        Arguments:
            metrics (list): List of metrics
            output (torch.Tensor): The output returned from the forward function
            batch (torch.Tensor): The batch processed with the prepare_batch function
            model (torch.nn.Module): The pytorch model
        """

        if self.compute_metrics_func is not None:
            return self.compute_metrics_func(metrics, output, batch, model=model)
        else:
            return self.do_compute_metrics(metrics, output, batch, model=model)

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

    def do_compute_losses(self, targets, output, batch):
        losses = []

        for target in targets:
            if target.output_index >= 0:
                output_data = output[target.output_index]
            else:
                output_data = output

            target_data = batch[target.target_index]

            loss = target.loss_fn(output_data, target_data) * target.weight
            losses.append(loss)

        return losses

    def do_compute_metrics(self, metrics, output, batch, model=None):
        metric_results = []

        for metric in metrics:
            if metric.output_index >= 0:
                output_data = output[metric.output_index].data
            else:
                output_data = output.data

            target_data = batch[metric.target_index].data

            result = metric.compute(output_data, target_data, model=model)
            metric_results.append(result)

        return metric_results
