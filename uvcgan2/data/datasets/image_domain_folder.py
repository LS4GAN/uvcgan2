import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from uvcgan2.consts import SPLIT_TRAIN

class ImageDomainFolder(Dataset):
    """Dataset structure introduced in a CycleGAN paper.

    This dataset expects images to be arranged into subdirectories
    under `path`: `trainA`, `trainB`, `testA`, `testB`. Here, `trainA`
    subdirectory contains training images from domain "a", `trainB`
    subdirectory contains training images from domain "b", and so on.

    Parameters
    ----------
    path : str
        Path where the dataset is located.
    domain : str
        Choices: 'a', 'b'.
    split : str
        Choices: 'train', 'test', 'val'
    transform : Callable or None,
        Optional transformation to apply to images.
        E.g. torchvision.transforms.RandomCrop.
        Default: None
    """

    def __init__(
        self, path,
        domain        = 'a',
        split         = SPLIT_TRAIN,
        transform     = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        subdir = split + domain.upper()

        self._path      = os.path.join(path, subdir)
        self._imgs      = ImageDomainFolder.find_images_in_dir(self._path)
        self._transform = transform

    @staticmethod
    def find_images_in_dir(path):
        extensions = set(IMG_EXTENSIONS)

        result = []
        for fname in os.listdir(path):
            fullpath = os.path.join(path, fname)

            if not os.path.isfile(fullpath):
                continue

            ext = os.path.splitext(fname)[1]
            if ext not in extensions:
                continue

            result.append(fullpath)

        result.sort()
        return result

    def __len__(self):
        return len(self._imgs)

    def __getitem__(self, index):
        path   = self._imgs[index]
        result = default_loader(path)

        if self._transform is not None:
            result = self._transform(result)

        return result

