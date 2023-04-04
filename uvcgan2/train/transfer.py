import os
import logging

import torch

from uvcgan2.consts import ROOT_OUTDIR
from uvcgan2.config import Args
from uvcgan2.cgan   import construct_model

LOGGER = logging.getLogger('uvcgan2.train')

def load_base_model(model, transfer_config):
    try:
        model.load(epoch = None)
        return

    except IOError as e:
        if not transfer_config.allow_partial:
            raise IOError(
                "Failed to find fully trained model in '%s' for transfer: %s"\
                % (model.savedir, e)
            ) from e

    LOGGER.warning(
        (
            "Failed to find fully trained model in '%s' for transfer."
            " Trying to load from a checkpoint..."
        ), model.savedir
    )

    epoch = model.find_last_checkpoint_epoch()

    if epoch > 0:
        LOGGER.warning("Load transfer model from a checkpoint '%d'", epoch)
    else:
        raise RuntimeError("Failed to find transfer model checkpoints.")

    model.load(epoch)

def get_base_model(transfer_config, device):
    base_path = os.path.join(ROOT_OUTDIR, transfer_config.base_model)
    base_args = Args.load(base_path)

    model = construct_model(
        base_args.savedir, base_args.config, is_train = True, device = device
    )

    load_base_model(model, transfer_config)

    return model

def transfer_from_larger_model(module, state_dict, strict):
    source_keys = set(state_dict.keys())
    target_keys = set(k for (k, _p) in module.named_parameters())

    matching_dict = { k : state_dict[k] for k in target_keys }
    module.load_state_dict(matching_dict, strict = strict)

    LOGGER.warning(
        "Transfer from a large model. Transferred %d / %d parameters",
        len(target_keys), len(source_keys)
    )

def collect_keys_for_transfer_to_wider_model(module, state_dict, strict):
    source_keys   = set(state_dict.keys())
    target_keys   = set()
    matching_keys = set()
    wider_keys    = set()

    for (k, p_target) in module.state_dict().items():
        target_keys.add(k)

        if strict:
            assert k in source_keys

        p_source = state_dict[k]

        shape_source = p_source.shape
        shape_target = p_target.shape

        if (shape_target is None) or (shape_target == shape_source):
            matching_keys.add(k)

        elif (
                (len(shape_target) == len(shape_source))
            and all(t >= s for (t, s) in zip(shape_target, shape_source))
        ):
            LOGGER.warning(
                "Transfer to wide model. Found wider parameter: '%s'."
                "%s vs %s",
                k, shape_target, shape_source
            )

            wider_keys.add(k)

        else:
            raise ValueError(
                "Transfer to wide model. "
                f"Cannot transfer parameter '{k}' due to mismatching"
                f"shapes {shape_target} vs {shape_source}"
            )

    if strict and (source_keys != target_keys):
        keys_diff = source_keys.symmetric_difference(target_keys)

        raise RuntimeError(
            "Transfer to wide model. Strict transfer failed due to"
            f" mismatching keys {keys_diff}"
        )

    return matching_keys, wider_keys

def transfer_to_wider_model(module, state_dict, strict):
    matching_keys, wider_keys = \
        collect_keys_for_transfer_to_wider_model(module, state_dict, strict)

    matching_dict = { k : state_dict[k] for k in matching_keys }
    module.load_state_dict(matching_dict, strict = False)

    for k, p in module.named_parameters():
        if k not in wider_keys:
            continue

        source_tensor = state_dict[k]
        target_slice  = tuple(slice(0, s) for s in source_tensor.shape)

        with torch.no_grad():
            p[target_slice] = source_tensor

def transfer_state_dict(module, state_dict, fuzzy, strict):
    if (fuzzy is None) or (fuzzy == 'none'):
        module.load_state_dict(state_dict, strict = strict)

    elif fuzzy == 'from-larger-model':
        transfer_from_larger_model(module, state_dict, strict)

    elif fuzzy == 'to-wider-model':
        transfer_to_wider_model(module, state_dict, strict)

    else:
        raise ValueError(f"Unknown fuzzy transfer type: {fuzzy}")

def transfer_parameters(model, base_model, transfer_config):
    for (dst, src) in transfer_config.transfer_map.items():
        transfer_state_dict(
            model.models[dst], base_model.models[src].state_dict(),
            transfer_config.fuzzy, transfer_config.strict
        )

def transfer(model, transfer_config):
    if transfer_config is None:
        return

    if isinstance(transfer_config, list):
        for conf in transfer_config:
            transfer(model, conf)
        return

    LOGGER.info(
        "Initiating parameter transfer : '%s'", transfer_config.to_dict()
    )

    base_model = get_base_model(transfer_config, model.device)
    transfer_parameters(model, base_model, transfer_config)


