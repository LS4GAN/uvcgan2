from uvcgan2.base.weight_init import init_weights
from uvcgan2.torch.funcs      import prepare_model
from uvcgan2.torch.spectr_norm import apply_sn

def default_model_init(model, model_config, device):
    model = prepare_model(model, device)
    init_weights(model, model_config.weight_init)

    if model_config.spectr_norm:
        apply_sn(model)

    return model

