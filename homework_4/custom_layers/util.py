def pipe_model(inputs, layers):
    """Pipes an input through a model to obtain the output hook"""

    for i in range(len(layers)):
        if i == 0:
            carry_out = layers[i](inputs)
        else:
            carry_out = layers[i](carry_out)

    return carry_out