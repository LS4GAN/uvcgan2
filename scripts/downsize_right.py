import argparse
import os
import tqdm
from PIL import Image

def downscale(image, out_shape, interpolation):
    if interpolation == 'bilinear':
        resample = Image.BILINEAR
    elif interpolation == 'bicubic':
        resample = Image.BICUBIC
    elif interpolation == 'lanczos':
        resample = Image.LANCZOS
    else:
        raise ValueError(f'Unknown interpolation method: {interpolation}')

    return image.resize(out_shape, resample = resample)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Resize Images with PIL')

    parser.add_argument(
        'source',
        help    = 'source directory',
        metavar = 'SOURCE',
        type    = str,
    )

    parser.add_argument(
        'target',
        help    = 'target directory',
        metavar = 'TARGET',
        type    = str,
    )

    parser.add_argument(
        '-s', '--shape',
        dest     = 'shape',
        help     = 'output shape',
        type     = int,
        required = True,
        nargs    = 2,
    )

    parser.add_argument(
        '-i',
        choices  = [ 'bilinear', 'bicubic', 'lanczos' ],
        default  = 'lanczos',
        dest     = 'interpolation',
        help     = 'interpolation',
        type     = str,
    )

    return parser.parse_args()

def collect_images(root):
    result = []

    for curr_root, _dirs, files in os.walk(root):
        rel_path = os.path.relpath(curr_root, root)

        for fname in files:
            result.append((rel_path, fname))

    return result

def load_image(path):
    result = Image.open(path)
    return result

def save_image(image, path):
    if not path.endswith('.png'):
        path = path + '.png'

    image.save(path)

def downscale_images(source, images, target, shape, interpolation):
    for (subdir, fname) in tqdm.tqdm(images, total = len(images)):
        path_src = os.path.join(source, subdir, fname)

        root_dst = os.path.join(target, subdir)
        path_dst = os.path.join(root_dst, fname)

        os.makedirs(root_dst, exist_ok = True)

        image_src = load_image(path_src)
        image_dst = downscale(image_src, shape, interpolation)

        save_image(image_dst, path_dst)

def main():
    cmdargs = parse_cmdargs()
    images = collect_images(cmdargs.source)

    downscale_images(
        cmdargs.source, images, cmdargs.target, cmdargs.shape,
        cmdargs.interpolation
    )

if __name__ == '__main__':
    main()

