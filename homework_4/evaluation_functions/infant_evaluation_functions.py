
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

def infant_test_func(*args):
	print("Evaluation worked for filename {}!".format(args[5]))

def create_ranked_filters(dataset, model, exp_cfg, revived_cfg, results, filename):

	# Get filter matrix
	filter_matrix = model.get_layer(name="conv").get_weights()[0]

	# Get one subject's data
	subject = dataset.load_subject(exp_cfg.dataset.train_subject_names[0])

	# Parse features from each week
	for key in subject.keys():
		subject[key] = subject[key][revived_cfg.dataset.feature_names]

	# Aggregate predictions for each week
	predictions = {}
	for key in subject.keys():
		predictions[key] = model.predict(np.expand_dims(subject[key].to_numpy(), axis=0))

	# Align ins, outs, and predictions for each week
	dataset.set_fields(revived_cfg)
	ins = np.stack( [subject[key] for key in sorted(subject.keys())] , axis=0)
	outs = np.stack( [dataset._label_week(dataset._parse_week_time(key)) for key in sorted(subject.keys())] , axis=0)
	preds = np.stack( [predictions[key] for key in sorted(predictions.keys())] , axis=0)

	# Plot
	fig, axs = plt.subplots(filter_matrix.shape[2], 2)
	for filter_ind in range(axs.shape[0]):
		# Plot filters with their channels in the first column
		for channel_ind in range(filter_matrix.shape[1]):
			axs[filter_ind, 0].plot(filter_matrix[:, channel_ind, filter_ind])

		# Plot labels and their corresponding predictions in the right column
		axs[filter_ind, 1].plot(outs)
		axs[filter_ind, 1].plot( preds[:, filter_ind] )
	plt.show()


def create_stacked_filter_responses(dataset, model, exp_cfg, revived_cfg, results, filename):
	
	# Get input data

	# Get filter response matrix

	# Plot
	fig, axs = plt.subplots(2, sharex=True, sharey=True)

def created_stacked_filter_responses_across_weeks(dataset, model, exp_cfg, revived_cfg, results, filename):
	
	# Get input data

	# Get filter response matrix

	# Plot
	fig, axs = plt.subplots(2, sharex=True, sharey=True)

def create_trajectory_responses(dataset, model, exp_cfg, revived_cfg, results, filename):
	
	# Get input data

	# Get filter response matrix

	# Plot
	fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

