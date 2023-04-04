import os

CONFIG_NAME = 'config.json'
ROOT_DATA   = os.environ.get('UVCGAN2_DATA',   'data')
ROOT_OUTDIR = os.environ.get('UVCGAN2_OUTDIR', 'outdir')

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

MERGE_PAIRED   = 'paired'
MERGE_UNPAIRED = 'unpaired'
MERGE_NONE     = 'none'

MODEL_STATE_TRAIN = 'train'
MODEL_STATE_EVAL  = 'eval'
