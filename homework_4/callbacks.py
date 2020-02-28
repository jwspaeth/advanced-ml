
import json
import copy

import tensorflow.keras.callbacks as keras_callbacks

class EarlyStopping(keras_callbacks.EarlyStopping):

    def __init__(self, exp_cfg, fbase=None):
        super().__init__(
                monitor="val_loss",
                patience=exp_cfg.callbacks.EarlyStopping.patience,
                restore_best_weights=True,
                min_delta=exp_cfg.callbacks.EarlyStopping.min_delta
            )

class FileMetricLogger(keras_callbacks.Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """

    def __init__(self, fbase, exp_cfg=None):
        super().__init__()
        self.file_path = fbase + "metric_logs.txt"
        with open(self.file_path, "w") as f:
            pass
        self.best_val_loss = None
        self.best_val_epoch = None
        self.best_val_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:

            file_logs = copy.deepcopy(logs)

            # Update best val loss and epoch
            if "val_loss" in file_logs.keys():
                if epoch == 0:
                    self.best_val_loss = file_logs["val_loss"]
                    self.best_val_epoch = epoch
                    for key, value in logs.items():
                        if "val" in key:
                            self.best_val_dict[key] = str(value)
                else:
                    if file_logs["val_loss"] < self.best_val_loss:
                        self.best_val_loss = file_logs["val_loss"]
                        self.best_val_epoch = epoch
                        for key, value in logs.items():
                            if "val" in key:
                                self.best_val_dict[key] = str(value)

            # Write state to file
            with open(self.file_path, "w") as f:
                if self.best_val_loss is not None:
                    f.write("Best val loss: {}\n".format(self.best_val_loss))
                    f.write("Best val dict: {}\n".format(json.dumps(self.best_val_dict)))
                    f.write("Best val epoch: {}\n\n".format(self.best_val_epoch))

                f.write("Epoch {}\n".format(epoch))
                for key, value in file_logs.items():
                    file_logs[key] = str(value)
                f.write(json.dumps(file_logs))