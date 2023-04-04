import torchvision
from torchvision import transforms

from uvcgan2.torch.select import extract_name_kwargs

TRANSFORM_DICT = {
    'center-crop'            : transforms.CenterCrop,
    'color-jitter'           : transforms.ColorJitter,
    'random-crop'            : transforms.RandomCrop,
    'random-flip-vertical'   : transforms.RandomVerticalFlip,
    'random-flip-horizontal' : transforms.RandomHorizontalFlip,
    'random-rotation'        : transforms.RandomRotation,
    'resize'                 : transforms.Resize,
    'CenterCrop'             : transforms.CenterCrop,
    'ColorJitter'            : transforms.ColorJitter,
    'RandomCrop'             : transforms.RandomCrop,
    'RandomVerticalFlip'     : transforms.RandomVerticalFlip,
    'RandomHorizontalFlip'   : transforms.RandomHorizontalFlip,
    'RandomRotation'         : transforms.RandomRotation,
    'Resize'                 : transforms.Resize,
}

def select_single_transform(transform):
    name, kwargs = extract_name_kwargs(transform)

    if name not in TRANSFORM_DICT:
        raise ValueError(f"Unknown transform: '{name}'")

    return TRANSFORM_DICT[name](**kwargs)

def select_transform(transform):
    result = []

    if transform is not None:
        if not isinstance(transform, (list, tuple)):
            transform = [ transform, ]

        result = [ select_single_transform(x) for x in transform ]

    result.append(torchvision.transforms.ToTensor())

    return torchvision.transforms.Compose(result)
