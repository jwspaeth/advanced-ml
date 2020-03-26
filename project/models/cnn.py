
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization

from custom_layers import pipe_model, hidden_stack, conv_stack_2d

def cnn(input_size, filters, kernels, strides, max_pool_sizes, cnn_l2,
    dnn_hidden_sizes, dnn_l2, n_options):
    """Construct a simple convolutional neural network"""

    inputs = Input(shape=input_size)

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

    intermediate = pipe_model(inputs, layers)

    # n_options is a list of actions with their discrete outputs.
    #   Length of list is number of simultenous actions
    #   Number in list is the number of values for that action
    output_layers = []
    for option_size in n_options:
        output_layers.append(
                    Dense(option_size)
            )

    # Pipe model by feeding through input placeholder
    outputs = []
    for output_layer in output_layers:
        outputs.append(
            pipe_model(intermediate, [output_layer])
            )

    return Model(inputs=inputs, outputs=outputs)


