import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from uvcgan2.consts import SPLIT_TRAIN
from .image_domain_folder import ImageDomainFolder

class ImageDomainHierarchy(Dataset):

    def __init__(
        self, path, domain,
        split     = SPLIT_TRAIN,
        transform = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._path      = os.path.join(path, split, domain)
        self._imgs      = ImageDomainFolder.find_images_in_dir(self._path)
        self._transform = transform

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        path   = self._imgs[index]
        result = default_loader(path)

        if self._transform is not None:
            result = self._transform(result)

        return result

