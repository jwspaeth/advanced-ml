
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization

from custom_layers import pipe_model, hidden_stack, conv_stack_2d

def cnn(input_size, filters, kernels, strides, max_pool_sizes, cnn_l2,
    dnn_hidden_sizes, dnn_l2, output_size):
    """Construct a simple convolutional neural network"""

    layers = []

    # Add conv layers with respective l2 values
    layers.append(
        conv_stack_2d(
                filters=filters,
                kernels=kernels,
                strides=strides,
                max_pool_sizes=max_pool_sizes,
                l2=cnn_l2
            )
        )

    # Flatten for dnn
    layers.append(
        Flatten()
        )

    # Add dnn
    layers.append(
        hidden_stack(
                hidden_sizes=dnn_hidden_sizes,
                l2=dnn_l2
            )
        )

    # Add output layer
    layers.append(
        Dense(
            output_size,
            )
        )

    # Pipe model by feeding through input placeholder
    inputs = Input(shape=input_size)
    outputs = pipe_model(inputs, layers)

    return Model(inputs=inputs, outputs=outputs)