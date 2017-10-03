import os

import torch

from .. import callback


class ModelCheckpointCallback(callback.Callback):
    """
    Stores the model on the filesystem at given points.

    Arguments:
        - path : Base path to store the model checkpoints.
        - after_num_epochs : Store the model after every [after_num_epochs] epochs.
        - after_num_batches : Store the model after every [after_num_batches] batches.
        - model_extraction_fn : If only part of the model should be stored. This function takes the full model as parameter and returns the part of the model that should be stored.

    """

    def __init__(self, path, after_num_epochs=1, after_num_batches=0, model_extraction_fn=None):
        super(ModelCheckpointCallback, self).__init__()

        self.after_num_epochs = after_num_epochs
        self.after_num_batches = after_num_batches
        self.model_extraction_fn = model_extraction_fn
        self.path = path

    def after_train_epoch(self, epoch_index, epoch_log):
        if self.after_num_epochs > 0 and (epoch_index + 1) % self.after_num_epochs == 0:
            self._store_model(epoch_index=epoch_index)

    def after_train_batch(self, epoch_index, batch_index, batch_log):
        if self.after_num_batches > 0 and (batch_index + 1) % self.after_num_batches == 0:
            self._store_model(epoch_index=epoch_index, batch_index=batch_index)

    def _store_model(self, epoch_index, batch_index=-1):
        model = self._trainer._model

        if self.model_extraction_fn is not None:
            model = self.model_extraction_fn(model)

        name = "epoch_{}".format(epoch_index)
        subfolder = ""

        if batch_index > -1:
            name = "batch_{}".format(batch_index)
            subfolder = "epoch_intermediates_{}".format(epoch_index)

        subfolder = os.path.join(self.path, subfolder)
        path = os.path.join(subfolder, name)

        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        torch.save(model.state_dict(), path)
