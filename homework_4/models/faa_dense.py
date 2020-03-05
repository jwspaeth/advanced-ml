
import copy

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization

from custom_layers import pipe_model, hidden_stack

def faa_dense(input_size, exp_cfg):
    """Construct a simple deep neural network"""

    layers = []

    layers.append(
            BatchNormalization(axis=2)
        )

    # Flatten input
    layers.append(
            Flatten()
        )

    # Add hidden layers with respective dropout and l2 values
    layers.append(
        hidden_stack(
            hidden_sizes=exp_cfg.model.hidden_sizes,
            batch_norms=exp_cfg.model.batch_norms,
            dropout=exp_cfg.model.dropout
            )
        )

    past = Dense(
                units=exp_cfg.model.output_size,
                activation=exp_cfg.model.output_act
                )

    future = Dense(
                units=exp_cfg.model.output_size,
                activation=exp_cfg.model.output_act
                )

    # Branch for both outputs
    inputs = Input(shape=input_size)
    intermediate = pipe_model(inputs, layers)

    # Use intermediate for both outputs
    past_output = pipe_model(intermediate, [past])
    future_output = pipe_model(intermediate, [future])

    return Model(inputs=inputs, outputs=[past_output, future_output])