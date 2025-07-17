def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = output.detach()
    return hook