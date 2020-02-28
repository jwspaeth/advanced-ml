from yacs.config import CfgNode as CN

# Construct default configuration
_D = CN()

# Save parameters
_D.save = CN()
_D.save.experiment_batch_name = "deep_test_1"

# Model parameters
_D.model = CN()
_D.model.input_axis_norm = 3
_D.model.conv = CN()
_D.model.conv.filters = [10, 20, 30, 40]
_D.model.conv.kernels = [3, 3, 3, 3]
_D.model.conv.strides = [2, 2, 2, 1]
_D.model.conv.l2 = 0
_D.model.conv.max_pool_sizes = [2, 2, 2, 1]
_D.model.conv.batch_norms = [1, 1, 1, 1]
_D.model.dense = CN()
_D.model.dense.hidden_sizes = [100, 25]
_D.model.dense.dropout = .5
_D.model.dense.batch_norms = [1, 1]
_D.model.output = CN()
_D.model.output.output_size = 2
_D.model.output.activation = "softmax"

# Training parameters
_D.train = CN()
_D.train.epochs = 200
_D.train.batch_size = 32
_D.train.loss = "mse"
_D.train.metrics = ["acc"]
_D.train.verbose = 2
_D.train.patience = 25
_D.train.min_delta = .0001

# Evaluation parameters
_D.evaluate = CN()

# Misc parameters
_D.misc = CN()
_D.misc.default_duplicate = 5 # Duplicates experiments by this amount. Only activates if all options are empty

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





