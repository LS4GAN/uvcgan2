import torch
from torch.nn.utils.parametrizations import spectral_norm

def apply_sn_to_module(module, name, n_power_iterations = 1):
    if isinstance(module, torch.nn.utils.parametrize.ParametrizationList):
        return

    if not hasattr(module, name):
        return

    w = getattr(module, name)

    if (w is None) or len(w.shape) < 2:
        return

    spectral_norm(module, name, n_power_iterations)

def apply_sn(module, tensor_name = 'weight', n_power_iterations = 1):
    submodule_list  = list(module.modules())

    for m in submodule_list:
        apply_sn_to_module(m, tensor_name, n_power_iterations)

