
import tensorflow.keras.backend as backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from custom_layers import pipe_model, hidden_stack

def dueling_dnn(input_size, hidden_sizes, n_options, hidden_act="elu", output_act="linear", dropout=0, l2=0):
    """Construct a simple deep neural network"""

    inputs = Input(shape=input_size)

    layers = []

    # Add hidden layers with respective dropout and l2 values
    layers.append(
        hidden_stack(hidden_sizes, hidden_act, dropout=dropout, l2=l2)
        )

    intermediate = pipe_model(inputs, layers)

    # Get V(s)
    state_values = Dense(1)(intermediate)

    # n_options is a list of actions with their discrete outputs.
    #   Length of list is number of simultenous actions
    #   Number in list is the number of values for that action
    output_layers = []
    for option_size in n_options:
        output_layers.append(
                    Dense(option_size, activation=output_act)
            )

    # Pipe model by feeding through input placeholder
    action_advantages = []
    for output_layer in output_layers:
        action_advantages.append(
            pipe_model(intermediate, [output_layer])
            )

    processed_advantages = [action_advantage - backend.max(action_advantage, axis=1, keepdims=True)
        for action_advantage in action_advantages]

    Q_values = [state_values + processed_advantage for processed_advantage in processed_advantages]

    return Model(inputs=inputs, outputs=Q_values)