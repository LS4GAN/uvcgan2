import logging
import random
import torch
import numpy as np

from torch import nn

LOGGER = logging.getLogger('uvcgan2.torch')

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_torch_device_smart():
    if torch.cuda.is_available():
        return 'cuda'

    return 'cpu'

def prepare_model(model, device):
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        LOGGER.warning(
            "Multiple (%d) GPUs found. Using Data Parallelism",
            torch.cuda.device_count()
        )
        model = nn.DataParallel(model)

    return model

def update_average_model(average_model, model, momentum):
    # NOTE: Perhaps this func needs to be rewritten w/o state_dicts.
    #       Currently, it works since Tensor.detach() returns a view
    #       of the data, instead of a copy.
    #       In the future, pytorch may modify this behavior.
    new_state = model.state_dict()

    with torch.no_grad():
        for (k, v) in average_model.state_dict().items():
            v[:] = momentum * v + (1 - momentum) * new_state[k]

