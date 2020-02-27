
import os
import importlib
from itertools import product

import exceptions
import models
import datasets
import callbacks as custom_callbacks

class config_handler:

	def __init__(self, config_name):
		self.config_name = config_name
		self.config_module = self._import_config(config_name)
		self._D = self.config_module._D
		self.all_options_dict = self.config_module.all_options_dict

	def get_default(self):
		"""Get a yacs CfgNode object with default values for my_project."""
		# Return a clone so that the defaults will not be altered
		# This is for the "local variable" use pattern
		return self._D.clone()

	def get_experiments(self):
		# Get cartesian product of options config applied to default config
		individual_options_list = self._get_individual_options_list()
		if len(individual_options_list) == 0:
			if self._D.misc.default_duplicate > 1:
				return [self._D.clone() for i in range(self._D.misc.default_duplicate)]
			else:
				return [self._D.clone()]

		exp_list = []
		for individual_option in individual_options_list:
			temp = self._D.clone()
			temp.merge_from_list(individual_option)
			exp_list.append(temp)

		return exp_list

	def get_experiment(self, experiment_num):
		# Fetch all experiments
		exp_cfg_list = self.get_experiments()

		# Index by number
		return exp_cfg_list[experiment_num]

	def get_option(self, experiment_num):
		# Fetch all individual options
		individual_option_list = self._get_individual_options_list()
		if len(individual_option_list) == 0:
			return {"default": None}

		# Index by number
		return individual_option_list[experiment_num]

	def get_num_experiments(self):
		# Fetch all experiments
		exp_cfg_list = self.get_experiments()

		# Return total number
		return len(exp_cfg_list)

	def get_options(self):
		# Fetch options
		return self.all_options_dict

	def get_dataset(self, exp_cfg):
		dataset_class = self._import_dataset(exp_cfg.dataset.name)
		dataset = dataset_class(exp_cfg.dataset)
		return dataset

	def get_model(self, input_size, exp_cfg):
		model_class = self._import_model(exp_cfg.model.name)
		model = model_class(input_size, exp_cfg)
		return model

	def get_callbacks(self, fbase, exp_cfg):
		callbacks = []
		for callback_name in exp_cfg.callbacks.names:
			callbacks.append( self._import_callback(callback_name)(fbase, exp_cfg) )

		return callbacks

	def _get_individual_options_list(self):
		cleaned_all_options_dict = self._clean_dict(self.all_options_dict)
		if len(list(cleaned_all_options_dict.keys())) == 0:
			return []

		options_dict_list = self._my_product(cleaned_all_options_dict)
		individual_options_list = [self._convert_dict_to_pair_list(options_dict) for options_dict in options_dict_list]

		return individual_options_list

	def _my_product(self, inp):
	    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]

	def _convert_dict_to_pair_list(self, in_dict):
		pair_list = []
		for key, value in in_dict.items():
			pair_list.append(key)
			pair_list.append(value)

		return pair_list

	def _clean_dict(self, in_dict):
		cleaned_dict = {}
		for key, value in in_dict.items():
			if len(in_dict[key]) != 0:
				cleaned_dict[key] = value

		return cleaned_dict

	def _import_config(self, config_name):
		try:
			config_module = importlib.import_module("config.{}".format(config_name))
		except Exception:
			raise exceptions.ConfigNotFoundException(config_name)

		return config_module

	def _import_dataset(self, dataset_name):
		# Find and import dataset from folder. If not found throw error.
		try:
			dataset_class = getattr(datasetss, dataset_name)
		except AttributeError:
			raise exceptions.DatasetNotFoundException(dataset_name)

		return model_class

	def _import_model(self, model_name):
		# Find and import model from module. If not found throw error.
		try:
			model_class = getattr(models, model_name)
		except AttributeError:
			raise exceptions.ModelNotFoundException(model_name)

		return model_class

	def _import_callback(self, callback_name):
		# Find and import callbacks from module. Existing keras callbacks must be wrapped in this module to
		# 	be explicitly called in the config file
		try:
			current_callback_class = getattr(custom_callbacks, callback_name)
		except AttributeError:
			raise exceptions.CallbackNotFoundException(callback_name)

		return current_callback_class



