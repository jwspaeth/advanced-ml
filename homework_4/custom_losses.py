
import tensorflow as tf
from tensorflow.keras import backend

def fan_mse(y_true, y_pred):
	# y_true will be (batch_size, 1)
	# y_pred will be (batch_size, num filters)
	# This function just squares the losses of each filter against the y_true, then averages
	running_sum = 0
	num_filters = backend.int_shape(y_pred)[1]
	for i in range( num_filters ):
		running_sum += (y_true - y_pred[:, i])**2
		#running_sum += backend.square(backend.subtract(y_true, y_pred[:, i]))
	return running_sum / num_filters