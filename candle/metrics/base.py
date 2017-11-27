class Metric(object):
    """
    A metric is used to compute some distance/measurement between output and target.
    The result of a metric can have multiple values/columns.

    Arguments:
        name (str): A name for identifying this metric.
        target_index (int): The index within a batch, where to find the target Tensor/Variable.
        output_index (int): The index to access the output Tensor/Variable in the output from the forward pass. If equal to -1 the output of the forward pass is used directly without indexing.
    """

    def __init__(self, name, target_index=1, output_index=-1):
        self.name = name
        self.target_index = target_index
        self.output_index = output_index

    def compute(self, output, target, model=None):
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
