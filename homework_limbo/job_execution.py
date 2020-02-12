#!/usr/bin/env python3

import sys

from config.configuration_handler import configuration_handler
from training.trainer import trainer

def execute_job():
	
	job_number = sys.argv[1]
	config_name = sys.argv[2]

	job_trainer = trainer(job_number, config_name)

if __name__ == "__main__":
	execute_job()