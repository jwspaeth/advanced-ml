from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras import regularizers

def cnn(input_size, exp_cfg):
    """Construct a simple convolutional neural network"""

    layers = []

    layers.append(
            BatchNormalization(axis=exp_cfg.model.input_axis_norm)
        )

    # Add conv layers with respective l2 values
    layers.append(
        conv_stack(
                filters=exp_cfg.model.conv.filters,
                kernels=exp_cfg.model.conv.kernels,
                strides=exp_cfg.model.conv.strides,
                max_pool_sizes=exp_cfg.model.conv.max_pool_sizes,
                batch_norms=exp_cfg.model.conv.batch_norms,
                l2=exp_cfg.model.conv.l2
            )
        )

    # Flatten for dnn
    layers.append(
        Flatten()
        )

    # Add dnn
    layers.append(
        hidden_stack(
                hidden_sizes=exp_cfg.model.dense.hidden_sizes,
                batch_norms=exp_cfg.model.dense.batch_norms,
                dropout=exp_cfg.model.dense.dropout
            )
        )

    # Add output layer
    layers.append(
        Dense(
            exp_cfg.model.output.output_size,
            activation=exp_cfg.model.output.activation
            )
        )

    # Pipe moel by feeding through input placeholder
    inputs = Input(shape=input_size)
    outputs = pipe_model(inputs, layers)

    return Model(inputs=inputs, outputs=outputs)

def dnn(input_size, hidden_sizes, output_size, hidden_act="elu", output_act="linear", dropout=0, l2=0):
    """Construct a simple deep neural network"""

    layers = []

    # Dropout layer if applicable
    if dropout > 0:
        layers.append(Dropout(rate=dropout))

    # Add hidden layers with respective dropout and l2 values
    layers.append(
        hidden_stack(hidden_sizes, hidden_act, dropout=dropout, l2=l2)
        )

    # l2 regularization if applicable
    if l2 > 0:
        layers.append(
            Dense(
                output_size,
                activation=output_act,
                kernel_regularizer=regularizers.l2(l2)
                )
            )
    else:
        layers.append(
            Dense(
                output_size,
                activation=output_act
                )
            )

    # Pipe model by feeding through input placeholder
    inputs = Input(shape=input_size)
    outputs = pipe_model(inputs, layers)

    return Model(inputs=inputs, outputs=outputs)

def conv_stack(filters, kernels, strides, max_pool_sizes, batch_norms=0, padding="same", activation="elu", l2=0):
    """Represents a stack of convolutional layers"""

    # If padding is one word or default, extend into a uniform list
    if type(batch_norms) != list:
        batch_norms = [batch_norms]*len(filters)

    # If padding is one word or default, extend into a uniform list
    if type(padding) != list:
        padding = [padding]*len(filters)

    # If activation is one word or default, extend into a uniform list
    if type(activation) != list:
        activation = [activation]*len(filters)

    layers = []
    for i in range(len(filters)):

        if l2 > 0:
            layers.append(Conv2D(
                            filters=filters[i],
                            kernel_size=kernels[i],
                            strides=strides[i],
                            padding=padding[i],
                            activation=activation[i],
                            kernel_regularizer=regularizers.l2(l2)
                ))
        else:
            layers.append(Conv2D(
                            filters=filters[i],
                            kernel_size=kernels[i],
                            strides=strides[i],
                            padding=padding[i],
                            activation=activation[i]
                ))

        if batch_norms[i] == 1:
            layers.append(BatchNormalization(axis=3))

        layers.append(MaxPool2D(pool_size=max_pool_sizes[i]))

    def conv_stack_layer(inputs):
        """Layer hook for stack"""

        return pipe_model(inputs, layers)

    return conv_stack_layer

def hidden_stack(hidden_sizes, batch_norms=0, hidden_act="elu", dropout=0, l2=0):
    """Represents a stack of neural layers"""

    if type(batch_norms) != list:
        batch_norms = [batch_norms]*len(hidden_sizes)

    layers = []
    for i, size in enumerate(hidden_sizes):

        # Apply l2 if applicable
        if l2 > 0:
            layers.append(Dense(
                    size, 
                    activation=hidden_act,
                    kernel_regularizer=regularizers.l2(l2)
                    )
                )
        else:
            layers.append(Dense(
                    size, 
                    activation=hidden_act,
                    )
                )

        if batch_norms[i] == 1:
            layers.append(BatchNormalization(axis=1))

        # Apply dropout if applicable
        if dropout > 0:
            layers.append(Dropout(rate=dropout))

    def hidden_stack_layer(inputs):
        """Layer hook for stack"""

        return pipe_model(inputs, layers)

    return hidden_stack_layer

def pipe_model(inputs, layers):
    """Pipes an input through a model to obtain the output hook"""

    for i in range(len(layers)):
        if i == 0:
            carry_out = layers[i](inputs)
        else:
            carry_out = layers[i](carry_out)

    return carry_out