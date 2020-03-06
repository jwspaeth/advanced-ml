
import os
import json
import copy

import tensorflow.keras.callbacks as keras_callbacks

class FileMetricLogger(keras_callbacks.Callback):
    """Logs metrics and best validation metrics to file"""

    def __init__(self, fbase, exp_cfg=None):
        super().__init__()
        fbase += "logs/"
        if not os.path.exists(fbase):
            os.mkdir(fbase)
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