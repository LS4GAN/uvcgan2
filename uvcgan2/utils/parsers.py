from uvcgan2.consts import (
    MODEL_STATE_TRAIN, MODEL_STATE_EVAL, SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL
)

def add_model_state_parser(parser):
    parser.add_argument(
        '--model-state',
        choices = [ MODEL_STATE_TRAIN, MODEL_STATE_EVAL ],
        default = 'eval',
        dest    = 'model_state',
        help    = "evaluate model in 'train' or 'eval' states",
        type    = str,
    )

def add_plot_extension_parser(parser, default = ( 'png', )):
    parser.add_argument(
        '-e', '--ext',
        default = None if default is None else list(default),
        dest    = 'ext',
        help    = 'plot extensions',
        type    = str,
        nargs   = '+',
    )

def add_batch_size_parser(parser, default = 1):
    parser.add_argument(
        '--batch-size',
        default = default,
        dest    = 'batch_size',
        help    = 'batch size to use for evaluation',
        type    = int,
    )

def add_n_eval_samples_parser(parser, default = None):
    parser.add_argument(
        '-n',
        default = default,
        dest    = 'n_eval',
        help    = 'number of samples to use for evaluation',
        type    = int,
    )

def add_eval_type_parser(parser, default = 'transfer'):
    parser.add_argument(
        '--type',
        choices = [ 'transfer', 'reco', 'masked', 'simple-reco' ],
        default = default,
        dest    = 'eval_type',
        help    = 'type of evaluation',
        type    = str,
    )

def add_split_parser(parser, default = SPLIT_TEST):
    parser.add_argument(
        '--split',
        choices = [ SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL ],
        default = default,
        dest    = 'split',
        help    = 'data split',
        type    = str,
    )

def add_eval_epoch_parser(parser, default = None):
    parser.add_argument(
        '--epoch',
        default = default,
        dest    = 'epoch',
        help    = (
            'checkpoint epoch to evaluate.'
            ' If not specified, then the evaluation will be performed for'
            ' the final model. If epoch is -1, then the evaluation will'
            ' be performed for the last checkpoint.'
        ),
        type    = int,
    )

def add_model_directory_parser(parser):
    parser.add_argument(
        'model',
        help    = 'directory containing model to evaluate',
        metavar = 'MODEL',
        type    = str,
    )

def add_preset_name_parser(
    parser, name, presets, default = None, help_msg = None,
):
    parser.add_argument(
        f'--{name}',
        default = default,
        dest    = name,
        choices = list(presets),
        help    = help_msg or name,
        type    = str,
    )

def add_standard_eval_parsers(
    parser,
    default_batch_size = 1,
    default_epoch      = None,
    default_n_eval     = None,
):
    add_model_directory_parser(parser)
    add_model_state_parser(parser)
    add_split_parser(parser)

    add_batch_size_parser(parser, default_batch_size)
    add_eval_epoch_parser(parser, default_epoch)
    add_n_eval_samples_parser(parser, default_n_eval)

