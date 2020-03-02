
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, MaxPool1D, AveragePooling1D, Lambda
from tensorflow.keras import backend

from custom_layers import pipe_model, conv_stack_1d

# To Do
#   • Incomplete: Rebuild visualizations
#   • Incomplete: Figure out why labels work despite mismatch
#   • Incomplete: activation regularization
#   • Incomplete: cross-filter regularization

def fan(input_size, exp_cfg):
    """Constructs the frequency analysis neural network"""

    layers = []

    # Tack on any remaining regularization to convolution layer
    #   • Incomplete: activation regularization
    #   • Incomplete: cross-filter regularization
    layers.append(
        conv_stack_1d(
                filters=exp_cfg.model.conv.filters,
                kernels=exp_cfg.model.conv.kernels,
                strides=exp_cfg.model.conv.strides,
                max_pool_sizes=exp_cfg.model.conv.max_pool_sizes,
                batch_norms=exp_cfg.model.conv.batch_norms,
                l2=exp_cfg.model.conv.l2,
                cross_filter_lambda=exp_cfg.model.conv.cross_filter_lambda,
                activation_lambda=exp_cfg.model.conv.activation_lambda,
                names=exp_cfg.model.conv.names
            )
        )

    # Max pooling layer
    layers.append(
            MaxPool1D(
                    pool_size=exp_cfg.model.max_pool.pool_size,
                    padding=exp_cfg.model.max_pool.padding
                )
        )

    # Average pooling layer
    layers.append(
            AveragePooling1D(
                    pool_size=exp_cfg.model.avg_pool.pool_size,
                    padding=exp_cfg.model.avg_pool.padding
                )
        )

    # Squeeze layer
    layers.append(
            Lambda(lambda x: backend.squeeze(x, axis=1))
        )

    # Rate modifier layer
    layers.append(
        Lambda(lambda x: x * exp_cfg.model.rate_modifier)
    )

    # Pipe model by feeding through input placeholder
    inputs = Input(shape=input_size)
    outputs = pipe_model(inputs, layers)

    return Model(inputs=inputs, outputs=outputs)