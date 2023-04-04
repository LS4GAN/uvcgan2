import logging

from uvcgan2.consts      import MERGE_PAIRED, MERGE_UNPAIRED, MERGE_NONE
from uvcgan2.utils.funcs import check_value_in_range

from .config_base import ConfigBase

LOGGER      = logging.getLogger('uvcgan2.config')
MERGE_TYPES = [ MERGE_PAIRED, MERGE_UNPAIRED, MERGE_NONE ]

class DatasetConfig(ConfigBase):
    """Dataset configuration.

    Parameters
    ----------
    dataset : str or dict
        Dataset specification.
    shape : tuple of int
        Shape of inputs.
    transform_train : None or str or dict or list of those
        Transformations to be applied to the training dataset.
        If `transform_train` is None, then no transformations will be applied
        to the training dataset.
        If `transform_train` is str, then its value is interpreted as a name
        of the transformation.
        If `transform_train` is dict, then it is expected to be of the form
        `{ 'name' : TRANFORM_NAME, **kwargs }`, where 'name' is the name of
        the transformation, and `kwargs` dict will be passed to the
        transformation constructor.
        Otherwise, `transform_train` is expected to be a list of values above.
        The corresponding transformations will be chained together in the
        order that they are specified.
        Default: None.
    transform_val : None or str or dict or list of those
        Transformations to be applied to the validation dataset.
        C.f. `transform_train`.
        Default: None.
    """

    __slots__ = [
        'dataset',
        'shape',
        'transform_train',
        'transform_test',
    ]

    def __init__(
        self, dataset, shape,
        transform_train = None,
        transform_test  = None,
    ):
        super().__init__()

        self.dataset         = dataset
        self.shape           = shape
        self.transform_train = transform_train
        self.transform_test  = transform_test

class DataConfig(ConfigBase):
    """Data configuration.

    Parameters
    ----------
    datasets : list of dict
        List of dataset specifications.
    merge_type : str, optional
        How to merge samples from datasets.
        Choices: 'paired', 'unpaired', 'none'.
        Default: 'unpaired'
    workers : int, optional
        Number of data workers.
        Default: None
    """

    __slots__ = [
        'datasets',
        'merge_type',
        'workers',
    ]

    def __init__(self, datasets, merge_type = MERGE_UNPAIRED, workers = None):
        super().__init__()

        check_value_in_range(merge_type, MERGE_TYPES, 'merge_type')
        assert isinstance(datasets, list)

        self.datasets    = [ DatasetConfig(**x) for x in datasets ]
        self.merge_type  = merge_type
        self.workers     = workers

def parse_deprecated_data_config_v1_celeba(
    dataset_args, image_shape, workers, transform_train, transform_val
):
    attr = dataset_args.get('attr', None)

    if attr is None:
        domains = [ None, ]
    else:
        domains = [ 'a', 'b' ]

    return DataConfig(
        datasets = [
            {
                'dataset' : {
                    'name'   : 'celeba',
                    'attr'   : attr,
                    'domain' : domain,
                    'path'   : dataset_args.get('path', None),
                },
                'shape'           : image_shape,
                'transform_train' : transform_train,
                'transform_test'  : transform_val,
            } for domain in domains
        ],
        merge_type = 'unpaired',
        workers    = workers,
    )

def parse_deprecated_data_config_v1_cyclegan(
    dataset_args, image_shape, workers, transform_train, transform_val
):
    return DataConfig(
        datasets = [
            {
                'dataset' : {
                    'name'   : 'cyclegan',
                    'domain' : domain,
                    'path'   : dataset_args.get('path', None),
                },
                'shape'           : image_shape,
                'transform_train' : transform_train,
                'transform_test'  : transform_val,
            } for domain in ['a', 'b']
        ],
        merge_type = 'unpaired',
        workers    = workers,
    )

def parse_deprecated_data_config_v1_imagedir(
    dataset_args, image_shape, workers, transform_train, transform_val
):
    return DataConfig(
        datasets = [
            {
                'dataset' : {
                    'name'   : 'imagedir',
                    'path'   : dataset_args.get('path', None),
                },
                'shape'           : image_shape,
                'transform_train' : transform_train,
                'transform_test'  : transform_val,
            },
        ],
        merge_type = 'none',
        workers    = workers,
    )

def parse_deprecated_data_config_v1_toyzero_precropped(
    dataset_args, image_shape, workers, transform_train, transform_val
):
    align_train = dataset_args.get('align_train', False)

    return DataConfig(
        datasets = [
            {
                'dataset' : {
                    'name'   : 'toyzero-precropped-v1',
                    'domain' : domain,
                    'path'   : dataset_args.get('path', None),
                },
                'shape'           : image_shape,
                'transform_train' : transform_train,
                'transform_test'  : transform_val,
            } for domain in [ 'a', 'b' ]
        ],
        merge_type = 'paired' if align_train else 'unpaired',
        workers    = workers,
    )

def parse_deprecated_data_config_v1_toyzero_presimple(
    dataset, dataset_args, image_shape, workers, transform_train, transform_val
):
    # pylint: disable=too-many-arguments
    if dataset == 'toyzero-presimple':
        merge_type = 'paired'

    elif dataset == 'toyzero-preunaligned':
        if dataset_args.get('align_train', False):
            merge_type = 'paired'
        else:
            merge_type = 'unpaired'

    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    return DataConfig(
        datasets = [
            {
                'dataset' : {
                    'name'    : 'toyzero-presimple-v1',
                    'domain'  : domain,
                    'fname'   : dataset_args.get('fname',   None),
                    'path'    : dataset_args.get('path',    None),
                    'seed'    : dataset_args.get('seed',    0),
                    'shuffle' : dataset_args.get('shuffle', True),
                },
                'shape'           : image_shape,
                'transform_train' : transform_train,
                'transform_test'  : transform_val,
            } for domain in [ 'a', 'b' ]
        ],
        merge_type = merge_type,
        workers    = workers,
    )

def parse_deprecated_data_config_v1(
    dataset, dataset_args, image_shape, workers,
    transform_train = None, transform_val = None
):
    # pylint: disable=too-many-arguments
    if dataset == 'celeba':
        return parse_deprecated_data_config_v1_celeba(
            dataset_args, image_shape, workers, transform_train, transform_val
        )

    if dataset == 'cyclegan':
        return parse_deprecated_data_config_v1_cyclegan(
            dataset_args, image_shape, workers, transform_train, transform_val
        )

    if dataset == 'imagedir':
        return parse_deprecated_data_config_v1_imagedir(
            dataset_args, image_shape, workers, transform_train, transform_val
        )

    if dataset == 'toyzero-precropped':
        return parse_deprecated_data_config_v1_toyzero_precropped(
            dataset_args, image_shape, workers, transform_train, transform_val
        )

    if dataset in [ 'toyzero-presimple', 'toyzero-preunaligned' ]:
        return parse_deprecated_data_config_v1_toyzero_presimple(
            dataset, dataset_args, image_shape, workers,
            transform_train, transform_val
        )

    raise NotImplementedError(
        f"Do not know how to parse deprecated '{dataset}'"
    )

def parse_data_config(data, data_args, image_shape, workers):
    if isinstance(data, str):
        LOGGER.warning(
            "Deprecation Warning: Old (v0) dataset configuration detected."
            " Please modify your configuration and change `data` parameter"
            " into a dictionary describing `DataConfig` structure."
        )
        return parse_deprecated_data_config_v1(
            data, data_args, image_shape, workers
        )

    assert data_args is None, \
        "Deprecated `data_args` argument detected with new data configuration"

    if (
           ('dataset' in data)
        or ('dataset_args' in data)
        or ('transform_train' in data)
        or ('transform_val' in data)
    ):
        LOGGER.warning(
            "Deprecation Warning: Old (v1) dataset configuration detected."
            " Please modify your configuration and change `data` parameter"
            " into a dictionary describing `DataConfig` structure."
        )
        return parse_deprecated_data_config_v1(
            **data, image_shape = image_shape, workers = workers
        )

    return DataConfig(**data)

