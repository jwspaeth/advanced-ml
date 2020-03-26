
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from custom_layers import pipe_model, hidden_stack

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