import argparse
import itertools
import os

import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models          import inception_v3, Inception_V3_Weights
from torchvision.datasets.folder import default_loader

import torchvision.transforms as T

import pandas as pd
import tqdm

class InceptionV3(nn.Module):

    def __init__(self, drop_last_layer = True, **kwargs):
        super().__init__()

        self.net = inception_v3(
            weights = Inception_V3_Weights.DEFAULT, **kwargs
        )

        if drop_last_layer:
            self.net.fc = nn.Identity()

    def forward(self, x):
        return self.net(x)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Eval inception v3 consistency metrics'
    )

    parser.add_argument(
        'root',
        help    = 'image directory',
        metavar = 'SOURCE',
        type    = str,
    )

    parser.add_argument(
        '-d', '--device',
        help    = 'device',
        default = 'cuda',
        dest    = 'device',
        type    = str,
    )

    return parser.parse_args()

@torch.no_grad()
def load_image(path, transforms, device):
    result = default_loader(path)
    result = transforms(result)
    return result.to(device).unsqueeze(0)

@torch.no_grad()
def calc_consistency_metrics(model, path_a, path_b, transforms, device):
    image_a = load_image(path_a, transforms, device)
    image_b = load_image(path_b, transforms, device)

    features_a = model(image_a)
    features_b = model(image_b)

    il1  = F.l1_loss(features_a, features_b).mean().item()
    il2  = torch.norm(features_a - features_b, p = 2).mean().item()
    icos = F.cosine_similarity(features_a, features_b).mean().item()

    return {
        'i-l1'  : il1,
        'i-l2'  : il2,
        'i-cos' : icos,
    }

def get_subdirs(root):
    result = []

    for subdir in os.listdir(root):
        full_path = os.path.join(root, subdir)
        if os.path.isdir(full_path):
            result.append(subdir)

    return result

def collect_images(root):
    return sorted(os.listdir(root))

def collect_subdir_pairs(subdirs):
    subdirs = [ subdir for subdir in subdirs if subdir.find('_hc_') == -1 ]

    source = [ subdir for subdir in subdirs if subdir.startswith('real_') ]
    target = [ subdir for subdir in subdirs if subdir.startswith('fake_') ]

    pairs = [
        (a, b) for (a, b) in itertools.product(source, target)
            if a[-2:] != b[-2:]
    ]

    return pairs

def save_metrics(root, label, metrics):
    metrics.to_csv(
        os.path.join(root, f'consist_metrics_inception_{label}.csv'),
        index = False
    )

def save_avg_metrics(root, metrics):
    metrics.to_csv(
        os.path.join(root, 'consist_metrics_inception.csv'), index = False
    )

def get_default_torch_transform():
    # NOTE: this can be found in the torch docs
    return T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def eval_consistency_metrics(root, device):
    # pylint: disable=too-many-locals
    result     = []
    subdirs    = get_subdirs(root)
    dir_pairs  = collect_subdir_pairs(subdirs)
    transforms = get_default_torch_transform()

    model = InceptionV3().to(device)
    model.eval()

    for (source_dir, target_dir) in dir_pairs:
        metrics = []
        root_source = os.path.join(root, source_dir)
        root_target = os.path.join(root, target_dir)

        images = collect_images(root_source)
        title  = f'Eval: {source_dir} vs {target_dir}'

        for fname in tqdm.tqdm(images, total = len(images), desc = title):
            path_source = os.path.join(root_source, fname)
            path_target = os.path.join(root_target, fname)

            curr_metrics = calc_consistency_metrics(
                model, path_source, path_target, transforms, device
            )
            curr_metrics['fname'] = fname
            metrics.append(curr_metrics)

        metrics = pd.DataFrame(metrics)
        save_metrics(root, f'{source_dir}_{target_dir}', metrics)

        avg_metrics = metrics.mean()
        avg_metrics['dir_a'] = source_dir
        avg_metrics['dir_b'] = target_dir

        result.append(avg_metrics.to_dict())

    return pd.DataFrame(result)

def main():
    cmdargs = parse_cmdargs()
    metrics = eval_consistency_metrics(cmdargs.root, cmdargs.device)

    save_avg_metrics(cmdargs.root, metrics)

if __name__ == '__main__':
    main()

