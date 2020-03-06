
import tensorflow.keras.callbacks as keras_callbacks

class EarlyStopping(keras_callbacks.EarlyStopping):
	"""Simple wrapper for configuration system"""

    def __init__(self, exp_cfg, fbase=None):
        super().__init__(
                monitor="val_loss",
                patience=exp_cfg.callbacks.EarlyStopping.patience,
                restore_best_weights=True,
                min_delta=exp_cfg.callbacks.EarlyStopping.min_delta
                )