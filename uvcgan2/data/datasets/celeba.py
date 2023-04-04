import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from uvcgan2.consts import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from uvcgan2.utils.funcs import check_value_in_range

FNAME_ATTRS = 'list_attr_celeba.txt'
FNAME_SPLIT = 'list_eval_partition.txt'
SUBDIR_IMG  = 'img_align_celeba'

SPLITS = {
    SPLIT_TRAIN : 0,
    SPLIT_VAL   : 1,
    SPLIT_TEST  : 2,
}

DOMAINS = [ 'a', 'b' ]

class CelebaDataset(Dataset):

    def __init__(
        self, path,
        attr      = 'Young',
        domain    = 'a',
        split     = SPLIT_TRAIN,
        transform = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        check_value_in_range(split, SPLITS, 'CelebaDataset: split')

        if attr is None:
            assert domain is None
        else:
            check_value_in_range(domain, DOMAINS, 'CelebaDataset: domain')

        super().__init__(**kwargs)

        self._path      = path
        self._root_imgs = os.path.join(path, SUBDIR_IMG)
        self._split     = split
        self._attr      = attr
        self._domain    = domain
        self._imgs      = []
        self._transform = transform

        self._collect_files()

    def _collect_files(self):
        imgs_specs = CelebaDataset.load_image_specs(self._path)

        imgs = CelebaDataset.partition_images(
            imgs_specs, self._split, self._attr, self._domain
        )

        self._imgs = [ os.path.join(self._root_imgs, x) for x in imgs ]

    @staticmethod
    def load_image_partition(root):
        path = os.path.join(root, FNAME_SPLIT)

        return pd.read_csv(
            path, sep = r'\s+', header = None, names = [ 'partition', ],
            index_col = 0
        )

    @staticmethod
    def load_image_attrs(root):
        path = os.path.join(root, FNAME_ATTRS)

        return pd.read_csv(
            path, sep = r'\s+', skiprows = 1, header = 0, index_col = 0
        )

    @staticmethod
    def load_image_specs(root):
        df_partition = CelebaDataset.load_image_partition(root)
        df_attrs     = CelebaDataset.load_image_attrs(root)

        return df_partition.join(df_attrs)

    @staticmethod
    def partition_images(image_specs, split, attr, domain):
        part_mask = (image_specs.partition == SPLITS[split])

        if attr is None:
            imgs = image_specs[part_mask].index.to_list()
        else:
            if domain == 'a':
                domain_mask = (image_specs[attr] > 0)
            else:
                domain_mask = (image_specs[attr] < 0)

            imgs = image_specs[part_mask & domain_mask].index.to_list()

        return imgs

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        path   = self._imgs[index]
        result = default_loader(path)

        if self._transform is not None:
            result = self._transform(result)

        return result

