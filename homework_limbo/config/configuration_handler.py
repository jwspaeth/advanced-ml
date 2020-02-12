class configuration_handler():

    experiment_path = "config/experiment/"
    config_path = "config/base/"

    def __init__(self, config_name):

        self.config_name = config_name

    def get_n_jobs(self):
        pass

    def get_hyperparameters(self, job_number):
        pass

    def get_base_configuration(self):
        pass

    def get_experiment_configuration(self, job_number):
        pass