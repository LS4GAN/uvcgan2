import argparse
import os

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.presets import GEN_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Pretrain CelebaHQ generators'
    )
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'uvcgan2')
    add_batch_size_parser(parser, default = 32)
    return parser.parse_args()

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'imagedir',
                    'path'   : 'celeba_hq_resized_lanczos',
                },
                'shape'           : (3, 256, 256),
                'transform_train' : [
                    'random-flip-horizontal',
                    { 'name' : 'random-rotation', 'degrees' : 10,  },
                    {
                        'name' : 'color-jitter',
                        'brightness' : 0.2,
                        'contrast'   : 0.2,
                        'saturation' : 0.2,
                        'hue'        : 0.2,
                    },
                ],
                'transform_test' : None,
            },
        ],
        'merge_type' : 'none',
        'workers'    : 1,
    },
    'epochs'        : 500,
    'discriminator' : None,
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-3 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'simple-autoencoder',
    'model_args' : {
        'masking' : {
            'name'       : 'image-patch-random',
            'patch_size' : (32, 32),
            'fraction'   : 0.4,
        },
    },
    'scheduler' : {
        'name'    : 'CosineAnnealingWarmRestarts',
        'T_0'     : 100,
        'T_mult'  : 1,
        'eta_min' : cmdargs.batch_size * 5e-8 / 512,
    },
    'loss'             : 'l1',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : f'pretrain-{cmdargs.gen}',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'celeba_hq_resized_lanczos'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 100,
}

train(args_dict)

