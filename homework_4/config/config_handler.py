
import os
import importlib
from itertools import product
import glob

from yacs.config import load_cfg
import tensorflow.keras.models as keras_models

import exceptions
import models
import datasets
import callbacks as custom_callbacks
import evaluation_functions
import custom_losses

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

    def get_mode(self):
        return self._D.mode

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

        mode = self._D.mode

        # If training mode, return config containing configuration
        # If eval mode, return relevant filename and default config
        if mode == "train":
            # Fetch all experiments
            exp_cfg_list = self.get_experiments()
            # Return total number
            return exp_cfg_list[experiment_num]
        elif mode == "eval":
            # Fetch all filenames matching regex
            filenames = glob.glob(self._D.evaluate.reload_path)
            return filenames[experiment_num], self._D.clone()

    def get_option(self, experiment_num):
        # Fetch all individual options
        individual_option_list = self._get_individual_options_list()
        if len(individual_option_list) == 0:
            return {"default": None}

        # Index by number
        return individual_option_list[experiment_num]

    def get_num_experiments(self):

        mode = self._D.mode

        # If training mode, number of experiments is number of combinations in the options dict
        # If eval mode, number of experiments is the number of models to reload and evaluate
        if mode == "train":
            # Fetch all experiments
            exp_cfg_list = self.get_experiments()
            # Return total number
            return len(exp_cfg_list)
        elif mode == "eval":
            # Fetch all filenames matching regex
            filenames = glob.glob(self._D.evaluate.reload_path)
            return len(filenames)

    def get_options(self):
        # Fetch options
        return self.all_options_dict

    def get_dataset(self, exp_cfg):
        dataset_class = self._import_dataset(exp_cfg.dataset.name)
        dataset = dataset_class(exp_cfg=exp_cfg)
        return dataset

    def get_model(self, exp_cfg=None, input_size=None, filename=None):

        if filename is not None: # For reloading model during evaluation
            revived_cfg = self.get_cfg_from_file(filename) # Get old cfg
            model = self.get_model(input_size=input_size, exp_cfg=revived_cfg) # Build model
            model.load_weights("{}model_and_cfg/saved_weights.h5".format(filename)) # Reload weights from file
            return model
        if exp_cfg.model.reload_path != "": # For reloading model architecture & weights from file
            revived_cfg = self.get_cfg_from_file(exp_cfg.model.reload_path) # Get old cfg
            model = self.get_model(input_size=input_size, exp_cfg=revived_cfg) # Build model
            model.load_weights("{}model_and_cfg/saved_weights.h5".format(exp_cfg.model.reload_path)) # Reload weights from file
            return model
        else: # For creating model from scratch using configuration file
            model_class = self._import_model(exp_cfg.model.name) # Import model
            model = model_class(input_size=input_size, exp_cfg=exp_cfg) # Build model
            return model

    def get_callbacks(self, fbase, exp_cfg):
        callbacks = []
        for callback_name in exp_cfg.callbacks.names:
            callbacks.append( self._import_callback(callback_name)(fbase=fbase, exp_cfg=exp_cfg) )

        return callbacks

    def get_evaluation_functions(self, exp_cfg):
        eval_functions = []
        for eval_name in exp_cfg.evaluate.evaluation_functions:
            eval_functions.append( self._import_evaluation_function(eval_name) )

        return eval_functions

    def get_loss(self, loss):
        if loss in dir(custom_losses):
            return getattr(custom_losses, loss)
        else:
            return loss

    def get_cfg_from_file(self, fbase):
        with open(fbase + "model_and_cfg/exp_cfg", "rt") as cfg_file:
            return load_cfg(cfg_file)

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
            dataset_class = getattr(datasets, dataset_name)
        except AttributeError:
            raise exceptions.DatasetNotFoundException(dataset_name)

        return dataset_class

    def _import_model(self, model_name):
        # Find and import model from module. If not found throw error.
        try:
            model_class = getattr(models, model_name)
        except AttributeError:
            raise exceptions.ModelNotFoundException(model_name)

        return model_class

    def _import_callback(self, callback_name):
        # Find and import callbacks from module. Existing keras callbacks must be wrapped in this module to
        #   be explicitly called in the config file
        try:
            current_callback_class = getattr(custom_callbacks, callback_name)
        except AttributeError:
            raise exceptions.CallbackNotFoundException(callback_name)

        return current_callback_class

    def _import_evaluation_function(self, func_name):
        try:
            eval_func = getattr(evaluation_functions, func_name)
        except AttributeError:
            raise exceptions.EvaluationFunctionNotFoundException(func_name)

        return eval_func



