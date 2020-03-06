"""
Various exceptions needed to run configuration system
"""

class MissingConfigArgException(Exception):

    def __init__(self):
        super().__init__("Configuration name missing from args. Use -cfg_name={config_name}")

class ConfigNotFoundException(Exception):

	def __init__(self, config_name):
		super().__init__("Config {} not found in config folder".format(config_name))

class DatasetNotFoundException(Exception):

	def __init__(self, dataset_name):
		super().__init__("Dataset {} not found in datasets folder".format(dataset_name))

class ModelNotFoundException(Exception):

	def __init__(self, model_name):
		super().__init__("Model {} not found in models folder".format(model_name))

class CallbackNotFoundException(Exception):

	def __init__(self, callback_name):
		super().__init__("Callback {} not found in callbacks folder".format(
			callback_name))

class EvaluationFunctionNotFoundException(Exception):

	def __init__(self, func_name):
		super().__init__("Evaluation function {} not found in evaluation_functions folder".format(func_name))