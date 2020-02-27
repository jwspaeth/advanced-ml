

class MissingConfigArgException(Exception):

    def __init__(self):
        super(MissingConfigArgException, self).__init__("Configuration name missing from args. Use -cfg_name={config_name}")

class ConfigNotFoundException(Exception):

	def __init__(self, config_name):
		super(ConfigNotFoundException, self).__init__("Config {} not found in config folder".format(config_name))

class DatasetNotFoundException(Exception):

	def __init__(self, dataset_name):
		super(DatasetNotFoundException, self).__init__("Dataset {} not found in dataset.py module".format(dataset_name))

class ModelNotFoundException(Exception):

	def __init__(self, model_name):
		super(ModelNotFoundException, self).__init__("Model {} not found in model.py module".format(model_name))

class CallbackNotFoundException(Exception):

	def __init__(self, callback_name):
		super(CallbackNotFoundException,
			self).__init__("Callback {} not found in callbacks.py or tensorflow.keras.callbacks modules".format(
			callback_name))