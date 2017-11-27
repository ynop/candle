import os

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.style.use('ggplot')


class TrainingLog(object):
    """
    Stores log data for a training run.

    Arguments:
        targets (list): list of targets that are going to be logged (index will be used for mapping).
        metrics (list): list of (metric, metric-name) tuples that are going to be logged (index will be used for mapping).

    Attributes:
        epochs (list): list of iteration logs for every epoch
    """

    def __init__(self, targets=[], metrics=[]):
        self.epochs = []

        self._targets = targets
        self._metrics = metrics

    def append_epoch_log(self, epoch_log):
        self.epochs.append(epoch_log)

    def save_in_folder(self, path):
        """ Save log data in the given folder. """
        for epoch_index, epoch in enumerate(self.epochs):
            epoch_folder = os.path.join(path, 'epoch_{}'.format(epoch_index))
            os.makedirs(epoch_folder, exist_ok=True)

            epoch.save_in_folder(epoch_folder)

    def save_loss_plots_at(self, path):
        """
        Save a plot as png for every loss function.
        """
        if len(self.epochs) <= 0:
            return

        dev_loss_x = np.arange(1, len(self.epochs) + 1)
        dev_losses_y = np.stack([np.array(epoch.dev_log.mean_loss()) for epoch in self.epochs], 1)

        train_loss_x_per_epoch = []
        train_losses_y_per_epoch = []

        for ep_index, epoch in enumerate(self.epochs):
            step = 1.0 / len(epoch.batches)
            train_loss_x_per_epoch.append(np.arange(step, 1.0 + step, step) + ep_index)
            train_losses_y_per_epoch.append(epoch.loss_values())

        train_loss_x = np.concatenate(train_loss_x_per_epoch)
        train_losses_y = np.concatenate(train_losses_y_per_epoch, 1)

        for i in range(np.size(train_losses_y, 0)):
            name = self._targets[i].name

            fig, ax = plt.subplots(figsize=(15, 13), dpi=80)
            ax.plot(train_loss_x, train_losses_y[i], label='Train')
            ax.plot(dev_loss_x, dev_losses_y[i], linewidth=2, label='Validation')

            ax.legend(loc='upper right')

            plt.savefig(os.path.join(path, '{}.pdf'.format(name)), bbox_inches='tight')

    def save_metric_plots_at(self, path):
        """
        Save a plot as png for every metric.
        """

        if len(self._metrics) <= 0 or len(self.epochs) <= 0:
            return

        dev = list(zip(*[epoch.dev_log.mean_metrics() for epoch in self.epochs]))
        train = list(zip(*[epoch.metric_values() for epoch in self.epochs]))
        train = [list(np.concatenate(x)) for x in train]

        dev_indices = np.arange(1, len(self.epochs) + 1)
        train_step = len(self.epochs) / len(train[0])
        train_indices = np.arange(train_step, len(self.epochs) + train_step, train_step)

        for index, metric in enumerate(self._metrics):
            fig, ax = plt.subplots(figsize=(15, 13), dpi=80)

            columns = metric.columns()

            for plot_column in metric.plotable_columns():
                column_index = columns.index(plot_column)

                train_data = np.stack(train[index]).T
                dev_data = np.stack(dev[index]).T

                if len(columns) == np.size(train_data, 0):
                    train_data = train_data[column_index]
                    dev_data = dev_data[column_index]

                ax.plot(train_indices, train_data, label='Train {}'.format(plot_column))
                ax.plot(dev_indices, dev_data, linewidth=2, label='Validation {}'.format(plot_column))

                ax.legend(loc='upper right')

            plt.savefig(os.path.join(path, '{}.pdf'.format(metric.name)), bbox_inches='tight')

    def write_stats_to(self, path):
        with open(path, 'w') as f:
            for index, epoch in enumerate(self.epochs):
                f.write('\n'.join([
                    "#" * 100,
                    "# EPOCH {}".format(index + 1),
                    "#" * 100,
                    ""
                ]))
                f.write(epoch.stats())
                f.write("\n")


class IterationLog(object):
    """
    Stores log data for one training or evaluation iteration.

    Arguments:
        targets (list): list of targets that are going to be logged (index will be used for mapping).
        metrics (list): list of (metric, metric-name) tuples that are going to be logged (index will be used for mapping).

    Attributes:
        batches (list): list of batch logs
        dev_log (IterationLog): Holds the evaluation log (for training iterations only)
    """

    __slots__ = ['dev_log', 'batches', '_targets', '_metrics']

    def __init__(self, targets=[], metrics=[]):
        self.dev_log = None
        self.batches = []

        self._targets = targets
        self._metrics = metrics

    def append_batch_log(self, batch_log):
        self.batches.append(batch_log)

    def loss_values(self):
        """
        Return a list of concatenated loss values. Every tuple in the list contains NUM_BATCHES values of one loss.

        e.g.
        [
            (0.2,0.19,0.15,...), # loss index 0
            (0.2,0.19,0.15,...), # loss index 1
            ...
        ]
        """
        return list(zip(*[x.loss for x in self.batches]))

    def metric_values(self):
        """
        Return a list of concatenated metric values. Every tuple in the list contains NUM_BATCHES values of one metric.

        e.g.
        [
            (3,4,56,6,4,4), # metric index 0
            ((3, 1),(4, 2),(56, 22),(6, 2),(4, 3),(4, 2)), # metric index 1
            ...
        ]
        """
        return list(zip(*[x.metrics for x in self.batches]))

    def mean_loss(self, recent=-1):
        """
        Return a list with means of all losses.

        Arguments:
            - recent : if > 0, use only [recent] number of batches.

        e.g.
        [
            0.2,    # loss 0
            0.32,   # loss 1
            ...
        ]
        """

        if recent > 0:
            values = np.array(self.loss_values())[:, -recent:]
        else:
            values = np.array(self.loss_values())

        return list(values.mean(1))

    def mean_metrics(self):
        """
        Return a list with cumulated metric values.

        e.g.
        [
            (4, 2, 0.5),    # metric 0
            0.34,           # metric 1
            ...
        ]
        """
        return [self._metrics[i].cumulate(x) for i, x in enumerate(self.metric_values())]

    def mean_loss_with_names(self, recent=-1):
        """
        Return a dict with means of losses with name as key.

        e.g.
        {
            "MSE" : 0.2,
            "BCE" : 0.32,
            ...
        }
        """
        target_names = [target.name for target in self._targets]

        return dict(zip(target_names, self.mean_loss(recent=recent)))

    def mean_metrics_with_names(self):
        """
        Return a dict with cumulated metrics with name as key.

        e.g.
        {
            "frame_accuracy" : (4, 2, 0.5),
            "squared_error" : 0.34,
            ...
        }
        """
        return dict(zip([x.name for x in self._metrics], self.mean_metrics()))

    def mean_metrics_with_objects(self):
        """
        Return a list with tuples ((metric_name, metric_instance), metric_mean).

        e.g.
        [
            (instance FrameMetric, (4, 2, 0.5)),
            (instance SquareMetric, 0.34),
            ...
        ]
        """
        return zip(self._metrics, self.mean_metrics())

    def save_in_folder(self, path):
        """ Save metrics and losses in the given directory. One loss/metric per file. """
        for index, values in enumerate(self.metric_values()):
            data = np.array(values)
            name = 'metric_{}'.format(self._metrics[index].name)

            np.save(os.path.join(path, name), data)

        for index, values in enumerate(self.loss_values()):
            data = np.array(values)
            name = 'loss_{}'.format(self._targets[index].name)

            np.save(os.path.join(path, name), data)

        if self.dev_log is not None:
            dev_log_path = os.path.join(path, 'dev_log')
            os.makedirs(dev_log_path, exist_ok=True)
            self.dev_log.save_in_folder(dev_log_path)

    def stats(self):
        """ Return a string with stats. """

        lines = []

        lines.append('LOSS')

        for name, mean in self.mean_loss_with_names().items():
            lines.append('  - {} : {}'.format(name, mean))

        lines.append('METRICS')

        for metric, metric_value in self.mean_metrics_with_objects():
            columns = metric.columns()

            if type(metric_value) not in (tuple, list):
                metric_value = [metric_value]

            stats_string = ' / '.join(['{} = {}'.format(columns[x], metric_value[x]) for x in range(len(columns))])
            lines.append('  - {} : {}'.format(metric.name, stats_string))

        if self.dev_log is not None:
            lines.append('VALIDATION')
            lines.extend(['     {}'.format(x) for x in self.dev_log.stats().split('\n')])

        lines.append("")

        return '\n'.join(lines)

    def write_stats_to(self, path):
        with open(path, 'w') as f:
            f.write(self.stats())


class BatchLog(object):
    """
    Stores loss and metric values for a single batch.

    The indices of the values have to correspond to the the loss/metric in the iteration log.

    Attributes:
        loss (list): List of captured loss values.
        metrics (list): List of captured metric values.
    """

    __slots__ = ['loss', 'metrics']

    def __init__(self):
        self.loss = []
        self.metrics = []
