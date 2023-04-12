import os
import math
from itertools import islice

from uvcgan2.config            import Args
from uvcgan2.consts            import (
    MODEL_STATE_TRAIN, MODEL_STATE_EVAL, MERGE_NONE
)
from uvcgan2.data              import construct_data_loaders
from uvcgan2.torch.funcs       import get_torch_device_smart, seed_everything
from uvcgan2.cgan              import construct_model

def slice_data_loader(loader, batch_size, n_samples = None):
    if n_samples is None:
        return (loader, len(loader))

    steps = min(math.ceil(n_samples / batch_size), len(loader))
    sliced_loader = islice(loader, steps)

    return (sliced_loader, steps)

def tensor_to_image(tensor):
    result = tensor.cpu().detach().numpy()

    if tensor.ndim == 4:
        result = result.squeeze(0)

    result = result.transpose((1, 2, 0))
    return result

def override_config(config, config_overrides):
    if config_overrides is None:
        return

    for (k,v) in config_overrides.items():
        config[k] = v

def get_evaldir(root, epoch, mkdir = False):
    if epoch is None:
        result = os.path.join(root, 'evals', 'final')
    else:
        result = os.path.join(root, 'evals', 'epoch_%04d' % epoch)

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def set_model_state(model, state):
    if state == MODEL_STATE_TRAIN:
        model.train()
    elif state == MODEL_STATE_EVAL:
        model.eval()
    else:
        raise ValueError(f"Unknown model state '{state}'")

def start_model_eval(path, epoch, model_state, merge_type, **config_overrides):
    args   = Args.load(path)
    device = get_torch_device_smart()

    override_config(args.config, config_overrides)
    args.config.data.merge_type = merge_type

    model = construct_model(
        args.savedir, args.config, is_train = False, device = device
    )

    if epoch == -1:
        epoch = max(model.find_last_checkpoint_epoch(), 0)

    print("Load checkpoint at epoch %s" % epoch)

    seed_everything(args.config.seed)
    model.load(epoch)

    set_model_state(model, model_state)
    evaldir = get_evaldir(path, epoch, mkdir = True)

    return (args, model, evaldir)

def load_eval_model_dset_from_cmdargs(
    cmdargs, merge_type = MERGE_NONE, **config_overrides
):
    args, model, evaldir = start_model_eval(
        cmdargs.model, cmdargs.epoch, cmdargs.model_state,
        merge_type = merge_type,
        batch_size = cmdargs.batch_size, **config_overrides
    )

    data_it = construct_data_loaders(
        args.config.data, args.config.batch_size, split = cmdargs.split
    )

    return (args, model, data_it, evaldir)

def get_eval_savedir(evaldir, prefix, model_state, split, mkdir = False):
    result = os.path.join(evaldir, f'{prefix}_{model_state}-{split}')

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def make_image_subdirs(model, savedir):
    for name in model.images:
        path = os.path.join(savedir, name)
        os.makedirs(path, exist_ok = True)

