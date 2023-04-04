from .config_base import ConfigBase

class TransferConfig(ConfigBase):
    """Model transfer configuration.

    Parameters
    ----------
    base_model : str
        Path to the model to transfer parameters from. If path is relative
        then this path is interpreted relative to `ROOT_OUTDIR`.
    transfer_map : dict or None
        Mapping between networks names of the current model and the model to
        transfer parameters from. For example, mapping of the form
        `{ 'gen_ab' : 'gen' }` will initialize generator `gen_ab` of the
        current model from the `gen` generator of the base model.
        Default: None.
    strict : bool
        Value of the pytorch's strict parameter when loading parameters.
        Default: True.
    allow_partial : bool
        Whether to allow transfer from the last checkpoint of a partially
        trained base model.
        Default: False.
    fuzzy : str, optional
        Allow fuzzy transfer. E.g. transfer parameters from a larger model to
        a smaller one.
        Choices: [ 'none', 'from-larger-model' ]
        Default: None.
    """

    __slots__ = [
        'base_model',
        'transfer_map',
        'strict',
        'allow_partial',
        'fuzzy',
    ]

    def __init__(
        self,
        base_model,
        transfer_map  = None,
        strict        = True,
        allow_partial = False,
        fuzzy         = None,
    ):
        # pylint: disable=too-many-arguments
        self.base_model    = base_model
        self.transfer_map  = transfer_map  or {}
        self.strict        = strict
        self.allow_partial = allow_partial
        self.fuzzy         = fuzzy

