# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan2/base/LICENSE` for copyright attribution and LICENSE

from torch.optim           import lr_scheduler
from uvcgan2.torch.select import extract_name_kwargs

def linear_scheduler(optimizer, epochs_warmup, epochs_anneal, verbose = True):

    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:
            return 1.0

        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)

    lr_fn = lambda epoch : lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn, verbose = verbose)

def get_scheduler(optimizer, scheduler):
    if scheduler is None:
        return None

    name, kwargs = extract_name_kwargs(scheduler)
    kwargs['verbose'] = True

    if name == 'linear':
        return linear_scheduler(optimizer, **kwargs)

    if name == 'step':
        return lr_scheduler.StepLR(optimizer, **kwargs)

    if name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if name == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)

    raise ValueError("Unknown scheduler '%s'" % name)

