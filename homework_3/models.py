from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers

def pipe_model(inputs, layers):
    """Pipes an input through a model to obtain the output hook"""

    for i in range(len(layers)):
        if i == 0:
            carry_out = layers[i](inputs)
        else:
            carry_out = layers[i](carry_out)

    return carry_out

def dnn(input_size, hidden_sizes, output_size, hidden_act="sigmoid", output_act="tanh", dropout=0, l2=0):
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

def hidden_stack(hidden_sizes, hidden_act="sigmoid", dropout=0, l2=0):
    """Represents a stack of neural layers"""

    layers = []
    for size in hidden_sizes:

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

        # Apply dropout if applicable
        if dropout > 0:
            layers.append(Dropout(rate=dropout))

    def hidden_stack_layer(inputs):
        """Layer hook for stack"""

        for i in range(len(layers)):
            if i == 0:
                carry_out = layers[i](inputs)
            else:
                carry_out = layers[i](carry_out)

        return carry_out

    return hidden_stack_layer