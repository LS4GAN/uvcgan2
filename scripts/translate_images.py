#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np
from PIL import Image

from uvcgan2.consts import MERGE_NONE
from uvcgan2.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir, make_image_subdirs
)
from uvcgan2.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions as images'
    )
    add_standard_eval_parsers(parser)
    add_plot_extension_parser(parser)

    return parser.parse_args()

def save_images(model, savedir, sample_counter, ext):
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        for index in range(torch_image.shape[0]):
            sample_index = sample_counter[name]

            image = tensor_to_image(torch_image[index])
            image = np.round(255 * image).astype(np.uint8)
            image = Image.fromarray(image)

            path  = os.path.join(savedir, name, f'sample_{sample_index}')
            for e in ext:
                image.save(path + '.' + e)

            sample_counter[name] += 1

def dump_single_domain_images(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter, ext
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
    desc = f'Translating domain {domain}'

    for batch in tqdm.tqdm(data_it, desc = desc, total = steps):
        model.set_input(batch, domain = domain)
        model.forward_nograd()

        save_images(model, savedir, sample_counter, ext)

def dump_images(model, data_list, n_eval, batch_size, savedir, ext):
    # pylint: disable=too-many-arguments
    make_image_subdirs(model, savedir)

    sample_counter = collections.defaultdict(int)
    if isinstance(ext, str):
        ext = [ ext, ]

    for domain, data_it in enumerate(data_list):
        dump_single_domain_images(
            model, data_it, domain, n_eval, batch_size, savedir,
            sample_counter, ext
        )

def main():
    cmdargs = parse_cmdargs()

    args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type = MERGE_NONE
    )

    if not isinstance(data_list, (list, tuple)):
        data_list = [ data_list, ]

    savedir = get_eval_savedir(
        evaldir, 'images', cmdargs.model_state, cmdargs.split
    )

    dump_images(
        model, data_list, cmdargs.n_eval, args.batch_size, savedir,
        cmdargs.ext
    )

if __name__ == '__main__':
    main()

