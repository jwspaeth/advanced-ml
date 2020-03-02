
import re
import glob
import sys
import math

import numpy as np
import pandas as pd

class InfantDataset:

    # Data path
    if "-s" in sys.argv:
        data_path = "/home/jwspaeth/datasets/baby1/"
    else:
        data_path = "/Users/willspaeth/datasets/baby1/"

    def __init__(self, exp_cfg=None):

        if exp_cfg is not None:
            self.set_fields(exp_cfg)
        else:
            self.shape = "sigmoid"
            self.offset = 0
            self.feature_names = []
            self.train_subject_names = []
            self.val_subject_names = []

    def set_fields(self, exp_cfg):
        self.shape = exp_cfg.dataset.shape
        self.offset = exp_cfg.dataset.offset
        self.feature_names = exp_cfg.dataset.feature_names
        self.train_subject_names = exp_cfg.dataset.train_subject_names
        self.val_subject_names = exp_cfg.dataset.val_subject_names

    def get_input_size(self):
        return (15000, len(self.feature_names))

    def load_data(self):
        """Loads data for training"""
        train_subjects_dict = self.load_subjects(self.train_subject_names)
        val_subjects_dict = self.load_subjects(self.val_subject_names)

        data_dict = {}
        data_dict["train"] = {
            "ins": [],
            "outs": []
        }

        if self.feature_names:
            for subject_name, week_dict in train_subjects_dict.items():
                for week_name, week_dataframe in week_dict.items():
                    data_dict["train"]["ins"].append( week_dataframe[self.feature_names].to_numpy() )
                    week_time = self._parse_week_time(week_name)
                    data_dict["train"]["outs"].append( self._label_week(week_time) )
        else:
            for subject_name, week_dict in train_subjects_dict.items():
                for week_name, week_dataframe in week_dict.items():
                    data_dict["train"]["ins"].append( week_dataframe.to_numpy() )
                    week_time = self._parse_week_time(week_name)
                    data_dict["train"]["outs"].append( self._label_week(week_time) )

        if val_subjects_dict:
            data_dict["val"] = {
                "ins": [],
                "outs": []
            }

            if self.feature_names:
                for subject_name, week_dict in val_subjects_dict.items():
                    for week_name, week_dataframe in week_dict.items():
                        data_dict["train"]["ins"].append( week_dataframe[self.feature_names].to_numpy() )
                        week_time = self._parse_week_time(week_name)
                        data_dict["train"]["outs"].append( self._label_week(week_time) )
            else:
                for subject_name, week_dict in val_subjects_dict.items():
                    for week_name, week_dataframe in week_dict.items():
                        data_dict["train"]["ins"].append( week_dataframe.to_numpy() )
                        week_time = self._parse_week_time(week_name)
                        data_dict["train"]["outs"].append( self._label_week(week_time) )

        if "val" in data_dict.keys():
            data_dict["val"]["ins"] = np.stack(data_dict["val"]["ins"], axis=0)
            data_dict["val"]["outs"] = np.stack(data_dict["val"]["outs"], axis=0)

        # Convert data dictionary entries into numpy arrays instead of lists
        data_dict["train"]["ins"] = np.stack(data_dict["train"]["ins"], axis=0)
        data_dict["train"]["outs"] = np.stack(data_dict["train"]["outs"], axis=0)

        return data_dict

    def load_subjects(self, subject_name_list):
        """Loads all subjects in the subject name list"""
        subjects_dict = {}
        for subject_name in subject_name_list:
            subjects_dict[subject_name] = self.load_subject(subject_name)

        return subjects_dict

    def load_all_subjects(self):
        """Loads all data, indexed by subject and weeks"""
        subject_names = self.get_available_subjects()

        all_subjects_dict = {}
        for subject_name in subject_names:
            all_subjects_dict[subject_name] = self.load_subject(subject_name)

        return all_subjects_dict

    def load_subject(self, subject_name):
        """Loads one subject's data, indexed by week"""
        filenames = glob.glob("{}*{}*".format(self.data_path, subject_name))
        if not filenames:
            raise Exception("Error: Subject not found.")

        subject_dict = {}
        for filename in filenames:
            week_key = re.search("w\d\d", filename).group(0)
            subject_dict[week_key] = pd.read_csv(filename)

        return subject_dict

    def load_week(self, subject_name, week_time):
        """Loads one week data"""
        filename = "{}subject_{}_w{:02d}.csv".format(self.data_path, subject_name, week_time)
        return pd.read_csv(filename)

    def load_weeks(self, week_time):
        """Loads all weeks with specific timestamp, regardless of subject"""
        filenames = glob.glob("{}*w{:02d}*".format(self.data_path, week_time))

        week_list = []
        for filename in filenames:
            week_list.append(pd.read_csv(filename))

        return week_list

    def get_available_subjects(self):
        filenames = glob.glob("{}*".format(self.data_path))
        subject_names = set()
        for filename in filenames:
            subject_name = re.search("(?<=_)*\w\d(?=_)", filename).group(0)
            subject_names.add(subject_name)

        return subject_names

    def _parse_week_time(self, week_name):
        digits = week_name.replace("w", "")
        return int(digits.lstrip("0"))

    def _label_week(self, week_time):
        """Labels weeks with their expected frequency, given a timestamp for that week"""

        if self.shape == "sigmoid":
            return self._sigmoid(self.offset, week_time)
        else:
            raise Exception("Error: unrecognized label shape. sigmoid currently supported")

    def _sigmoid(self, offset, week_time):
        
        x = week_time - offset
        y = 1 / (1 + (math.exp( (-1*(x) ) )))

        return y