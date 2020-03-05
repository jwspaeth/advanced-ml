
from yacs.config import CfgNode as CN
from custom_losses import fan_mse

# Construct root
_D = CN()

# Training or evaluation
_D.mode = "eval"

# Save parameters
_D.save = CN()
_D.save.experiment_batch_name = "fan_dummy_test"

# Dataset parameters
_D.dataset = CN()
_D.dataset.name = "InfantDataset"
_D.dataset.shape = "sigmoid"
_D.dataset.offset = 6
_D.dataset.feature_names = ["left_wrist_x"]
_D.dataset.train_subject_names = ["k3"]
_D.dataset.val_subject_names = []

# Model parameters
_D.model = CN()
_D.model.name = "fan"
_D.model.input_axis_norm = 2
_D.model.conv = CN()
_D.model.conv.filters = [3]
_D.model.conv.kernels = [25] # Half second window
_D.model.conv.strides = [1]
_D.model.conv.max_pool_sizes = [1]
_D.model.conv.batch_norms = [0]
_D.model.conv.l2 = 0
_D.model.conv.cross_filter_lambda = 0
_D.model.conv.activation_lambda = 0
_D.model.conv.names = ["conv"]
_D.model.max_pool = CN()
_D.model.max_pool.pool_size = (10 * 50) # Seconds * (timesteps/second)
_D.model.max_pool.padding = "same"
_D.model.avg_pool = CN()
_D.model.avg_pool.pool_size = 30
_D.model.avg_pool.padding = "same"
_D.model.rate_modifier = 2
# If reload is an empty string, doesn't do anything. Otherwise looks for the reload location.
# If reload location is found, it reloads the architecture and weights from file while forgetting
# all of the other parameters listed here. Location should only define path to batch and experiment
_D.model.reload_path = ""

# Training parameters
_D.train = CN()
_D.train.optimizer = "adam"
_D.train.epochs = 200
_D.train.batch_size = 32
_D.train.loss = "fan_mse"
_D.train.metrics = []
_D.train.verbose = 2

# Callback parameters
_D.callbacks = CN()
_D.callbacks.names = ["FileMetricLogger"]

# Evaluation parameters
_D.evaluate = CN()
_D.evaluate.reload_path = "results/fan_dummy_test/*/"
#_D.evaluate.evaluation_functions = ["test_func", "create_ranked_filters"]
_D.evaluate.evaluation_functions = ["infant_test_func"]

# Misc parameters
_D.misc = CN()
_D.misc.default_duplicate = 1 # Duplicates experiments by this amount. Only activates if all options are empty

# Construct list of configuration keys and their possible options
# • If the key is in the list, the default is overwritten, unless its corresponding value list is empty
all_options_dict = {
	"model.conv.filters": [],
	"model.conv.kernels": [],
	"model.conv.strides": [],
	"model.conv.l2": [],
	"model.dense": [],
	"model.dense.hidden_sizes": [],
	"model.dense.dropout": []
}






