
from config.configuration_handler import configuration_handler
from datasets.dataset_handling import import_dataset
from models.model_handling import import_model
from training.callback_handling import import_callbacks

class trainer():

    def __init__(self, job_number, config_name):

        self.config_handler = configuration_handler(config_name)
        self.hyperparameters = self.config_handler.get_hyperparameters(job_number)
        self.base_config = self.config_handler.get_base_configuration()
        self.experiment_config = self.config_handler.get_experiment_configuration(job_number)

      	self.dataset = import_dataset(experiment_config.Data)
      	self.model = import_model(experiment_config.Model)
      	self.callbacks = import_callbacks(experiment_config.Callback)

    def train(self):

    	data = self.dataset.load_data()

	    history = model.fit(
			    		x=data["train"]["ins"],
			    		y=data["train"]["outs"],
			    		validation_data=(data["val"]["ins"], data["val"]["outs"]),
			    		batch_size=self.experiment_config.Training.batch_size,
			    		epochs=self.experiment_config.Training.epochs,
			    		verbose=self.experiment_config.Training.verbose,
			    		callbacks=self.callbacks
			    		)

	    