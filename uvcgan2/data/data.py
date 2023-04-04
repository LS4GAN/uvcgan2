import os
import torch

import torchvision

from uvcgan2.consts       import (
    ROOT_DATA, SPLIT_TRAIN, MERGE_PAIRED, MERGE_UNPAIRED
)
from uvcgan2.torch.select import extract_name_kwargs

from .datasets.celeba                 import CelebaDataset
from .datasets.image_domain_folder    import ImageDomainFolder
from .datasets.image_domain_hierarchy import ImageDomainHierarchy
from .datasets.zipper                 import DatasetZipper

from .loader_zipper import DataLoaderZipper
from .transforms    import select_transform

def select_dataset(name, path, split, transform, **kwargs):
    if name == 'celeba':
        return CelebaDataset(
            path, transform = transform, split = split, **kwargs
        )

    if name in [ 'cyclegan', 'image-domain-folder' ]:
        return ImageDomainFolder(
            path, transform = transform, split = split, **kwargs
        )

    if name in [ 'image-domain-hierarchy' ]:
        return ImageDomainHierarchy(
            path, transform = transform, split = split, **kwargs
        )

    if name == 'imagenet':
        return torchvision.datasets.ImageNet(
            path, transform = transform, split = split, **kwargs
        )

    if name in [ 'imagedir', 'image-folder' ]:
        return torchvision.datasets.ImageFolder(
            os.path.join(path, split), transform = transform, **kwargs
        )

    raise ValueError(f"Unknown dataset: {name}")

def construct_single_dataset(dataset_config, split):
    name, kwargs = extract_name_kwargs(dataset_config.dataset)
    path         = os.path.join(ROOT_DATA, kwargs.pop('path', name))

    if split == SPLIT_TRAIN:
        transform = select_transform(dataset_config.transform_train)
    else:
        transform = select_transform(dataset_config.transform_test)

    return select_dataset(name, path, split, transform, **kwargs)

def construct_datasets(data_config, split):
    return [
        construct_single_dataset(config, split)
            for config in data_config.datasets
    ]

def construct_single_loader(
    dataset, batch_size, shuffle,
    workers         = None,
    prefetch_factor = 20,
    **kwargs
):
    if workers is None:
        workers = min(torch.get_num_threads(), 20)

    return torch.utils.data.DataLoader(
        dataset, batch_size,
        shuffle         = shuffle,
        num_workers     = workers,
        prefetch_factor = prefetch_factor,
        pin_memory      = True,
        **kwargs
    )

def construct_data_loaders(data_config, batch_size, split):
    datasets = construct_datasets(data_config, split)
    shuffle  = (split == SPLIT_TRAIN)

    if data_config.merge_type == MERGE_PAIRED:
        dataset = DatasetZipper(datasets)

        return construct_single_loader(
            dataset, batch_size, shuffle, data_config.workers,
            drop_last = False
        )

    loaders = [
        construct_single_loader(
            dataset, batch_size, shuffle, data_config.workers,
            drop_last = (data_config.merge_type == MERGE_UNPAIRED)
        ) for dataset in datasets
    ]

    if data_config.merge_type == MERGE_UNPAIRED:
        return DataLoaderZipper(loaders)

    if len(loaders) == 1:
        return loaders[0]

    return loaders

