class Metric(object):
    """
    A metric is used to compute some distance/measurment between output and target.

    The result of a metric can have multiple values/columns.
    """

    def compute(self, output, batch):
        """ Return the computed metrics as single number or list/tuple of values. """
        pass

    def cumulate(self, metric_values=[]):
        """
        Cumulate a list of metric values (returned by compute).

        e.g. sum of total and correct labels
        input : [ [4,1], [5,2], [5,4], [6,3] ]
        output : [ 20, 10]
        """
        pass

    @classmethod
    def plotable_columns(cls):
        """
        Return the names of the columns that can be used to plot.

        e.g. [ratio]
        """
        return cls.columns()

    @classmethod
    def columns(cls):
        """
        Return the labels for the columens.

        e.g. When a metric computes the number of total and correct labels and a ratio of correct labels.

        [ "total", "correct", "ratio"]
        """
        raise NotImplementedError("Metrics have to implement columns().")

