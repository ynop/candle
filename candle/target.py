class Target(object):
    """
    Represents a loss function that is computed for a given output/target pair.

    Arguments:
        - name : A name for identifying this target.
        - loss_function : A pytorch loss function.
        - target_index : The index within a batch, where to find the target Tensor/Variable.
        - output_index : The index to access the output Tensor/Variable in the output from the forward pass. If equal to -1 the output of the forward pass is used directly without indexing.

    """

    def __init__(self, name, loss_function, target_index=1, output_index=-1):
        self.name = name
        self.loss_fn = loss_function
        self.target_index = target_index
        self.output_index = output_index
