
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from custom_layers import pipe_model, hidden_stack

def dnn(input_size, hidden_sizes, n_options, hidden_act="elu", output_act="linear", dropout=0, l2=0, **kwargs):
    """Construct a simple deep neural network"""

    inputs = Input(shape=input_size)

    layers = []

    # Dropout layer if applicable
    if dropout > 0:
        layers.append(Dropout(rate=dropout))

    # Add hidden layers with respective dropout and l2 values
    layers.append(
        hidden_stack(hidden_sizes, hidden_act=hidden_act, dropout=dropout, l2=l2)
        )

    intermediate = pipe_model(inputs, layers)

    # n_options is a list of actions with their discrete outputs.
    #   Length of list is number of simultenous actions
    #   Number in list is the number of values for that action
    output_layers = []
    for option_size in n_options:
        output_layers.append(
                    Dense(option_size, activation=output_act)
            )

    # Pipe model by feeding through input placeholder
    outputs = []
    for output_layer in output_layers:
        outputs.append(
            pipe_model(intermediate, [output_layer])
            )

    return Model(inputs=inputs, outputs=outputs)