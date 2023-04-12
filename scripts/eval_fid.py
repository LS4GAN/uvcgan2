import argparse
import os
import pandas as pd

import torch_fidelity

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Evaluate Perceptual Metrics'
    )

    parser.add_argument(
        'root',
        help    = 'directory with images',
        metavar = 'ROOT',
        type    = str,
    )

    parser.add_argument(
        '--kid-size',
        default = 100,
        dest    = 'kid_size',
        help    = 'size of the KID subsets',
        type    = int,
    )

    return parser.parse_args()

def get_subdirs(root):
    result = list(sorted(
        x for x in os.listdir(root)
            if os.path.isdir(os.path.join(root, x))
    ))

    result = [ x for x in result if (x.find('_hc_')   == -1) ]
    result = [ x for x in result if (x.find('_mask_') == -1) ]

    return result

def get_ref_subdirs(subdirs):
    result = []
    REF_PREFIX = [ 'real' ]

    for s in subdirs:
        for p in REF_PREFIX:
            if s.startswith(p):
                result.append(s)
                break

    return result

def get_test_subdirs(subdirs):
    result = []
    TEST_PREFIX = [ 'fake', 'reco' ]

    for s in subdirs:
        for p in TEST_PREFIX:
            if s.startswith(p):
                result.append(s)
                break

    return result

def evaluate_metrics(path1, path2, kid_size):
    return torch_fidelity.calculate_metrics(
        input1  = path1,
        input2  = path2,
        cuda    = True,
        isc     = False,
        fid     = True,
        kid     = True,
        verbose = False,
        kid_subset_size = kid_size,
    )

def get_suffix(s):
    tokens = s.split('_')
    if len(tokens) < 2:
        return ''

    return tokens[-1]

def match_suffix(a, b):
    suff_a = get_suffix(a)
    suff_b = get_suffix(b)

    return (suff_a == suff_b)

def evaluate_metrics_matrix(root, subdirs, kid_size):
    result = []

    for a in get_ref_subdirs(subdirs):
        for b in get_test_subdirs(subdirs):
            if not match_suffix(a, b):
                continue

            print(f"Calculating metrics: {a} <-> {b}")
            metrics = evaluate_metrics(
                os.path.join(root, a), os.path.join(root, b), kid_size
            )
            metrics['dir_a'] = a
            metrics['dir_b'] = b

            result.append(metrics)

    return pd.DataFrame(result)

def save_metrics(root, metrics):
    metrics.to_csv(os.path.join(root, 'fid_metrics.csv'), index = False)

def main():
    cmdargs = parse_cmdargs()

    subdirs = get_subdirs(cmdargs.root)
    metrics = evaluate_metrics_matrix(cmdargs.root, subdirs, cmdargs.kid_size)

    save_metrics(cmdargs.root, metrics)

if __name__ == '__main__':
    main()

