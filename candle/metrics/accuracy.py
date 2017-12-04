from . import base

import numpy as np


class BinaryAccuracy(base.Metric):
    def compute(self, output, target, model=None):
        output_classes = output.round()
        target_classes = target.round()

        cmp = target_classes.eq(output_classes)

        total = cmp.numel()
        correct = cmp.sum()

        return total, correct, correct / total

    def cumulate(self, metric_values=[]):
        data = np.stack(metric_values).T

        total = data[0].sum()
        correct = data[1].sum()
        accuracy = correct / total

        return total, correct, accuracy

    @classmethod
    def plotable_columns(cls):
        return ["accuracy"]

    @classmethod
    def columns(cls):
        return ["total", "correct", "accuracy"]


class CategoricalAccuracy(base.Metric):
    def compute(self, output, target, model=None):
        output_maxes = output.topk(1, 1)[1]
        target_maxes = target.topk(1, 1)[1]

        cmp = target_maxes.eq(output_maxes)

        total = cmp.size(0)
        correct = cmp.sum()

        return total, correct, correct / total

    def cumulate(self, metric_values=[]):
        data = np.stack(metric_values).T

        total = data[0].sum()
        correct = data[1].sum()
        accuracy = correct / total

        return total, correct, accuracy

    @classmethod
    def plotable_columns(cls):
        return ["accuracy"]

    @classmethod
    def columns(cls):
        return ["total", "correct", "accuracy"]
