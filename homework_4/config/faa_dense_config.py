
from yacs.config import CfgNode as CN

# Construct root
_D = CN()

# Training or evaluation
_D.mode = "train"

# Save parameters
_D.save = CN()
_D.save.experiment_batch_name = "faa_test_6"

# Dataset parameters
_D.dataset = CN()
_D.dataset.name = "FaaDataset"
_D.dataset.feature_length = 6

# Model parameters
_D.model = CN()
_D.model.name = "faa_dense"
_D.model.input_axis_norm = 2
_D.model.hidden_sizes = [20, 10, 10, 10]
_D.model.batch_norms = [1, 1, 1, 1]
_D.model.dropout = 0
_D.model.output_size = 3
_D.model.output_act = "linear"
# If reload is an empty string, doesn't do anything. Otherwise looks for the reload location.
# If reload location is found, it reloads the architecture and weights from file while forgetting
# all of the other parameters listed here. Location should only define path to batch and experiment
_D.model.reload_path = ""

# Training parameters
_D.train = CN()
_D.train.learning_rate = .0001
_D.train.epochs = 200
_D.train.batch_size = 32
_D.train.loss = "mse"
_D.train.metrics = []
_D.train.verbose = 2

# Callback parameters
_D.callbacks = CN()
_D.callbacks.names = ["EarlyStopping", "FileMetricLogger"]
_D.callbacks.EarlyStopping = CN()
_D.callbacks.EarlyStopping.patience = 50
_D.callbacks.EarlyStopping.min_delta = .0001

# Evaluation parameters
_D.evaluate = CN()
_D.evaluate.reload_path = "results/faa_test_4/*/"
_D.evaluate.evaluation_functions = ["val_learning_curve", "roll_forward", "roll_backward"]
_D.evaluate.rollforward_length = 20
_D.evaluate.rollback_length = 20
_D.evaluate.n_rollout_snapshots = 10

# Misc parameters
_D.misc = CN()
_D.misc.default_duplicate = 1 # Duplicates experiments by this amount. Only activates if all options are empty

# Construct list of configuration keys and their possible options
# • If the key is in the list, the default is overwritten, unless its corresponding value list is empty
all_options_dict = {}






