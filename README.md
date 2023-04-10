# Overview

This package provides reference implementation of the `Rethinking CycleGAN:
Improving Quality of GANs for Unpaired Image-to-Image Translation`
[paper][uvcgan2_paper].

`uvcgan2` builds upon the CycleGAN method for unpaired image-to-image transfer
and imporoves its performance by modifyng the generator, discriminator, and the
training procedure.

This README file provides brief instructions about how to set up the `uvcgan2`
package and reproduce the paper results. To facilitate the XXXX we share
the pre-trained models (c.f. SECTION_PRETRAINED).

The code of `uvcgan2` is based on [pytorch-CycleGAN-and-pix2pix][cyclegan_repo]
and [uvcgan][uvcgan_repo]. Please refer to the LICENSE section for the proper
copyright attribution.


# Installation & Requirements

## Requirements

`uvcgan2` models were trained under the official `pytorch` container
`pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`. A similar training
environment can be constructed with `conda`
```
conda env create -f contrib/conda_env.yaml
```

## Installation

To install the `uvcgan2` package one can simply run the following command
```
python3 setup.py develop --user
```
from the `uvcgan2` source tree.

## Environment Setup

`uvcgan2` extensively uses two environment variables `UVCGAN2_DATA` and
`UVCGAN2_OUTDIR` to locate user data and output directories. Users are advised
to set these environment variables. `uvcgan2` will look for datasets in the
`${UVCGAN2_DATA}` directory and will save results under the `${UVCGAN2_OUTDIR}`
directory. If these variables are not set then they will default to `./data`
and `./outdir` correspondingly.


# UVCGANv2 Reproduction

To reproduce the results of the paper, the following workflow is suggested:

1. Download datasets (`selfie2anime`, `celeba`, `celeba_hq`, `afhq`).
2. Pre-process high-quality datasets.
3. Pre-train generators on an Inpainting pretext task.
4. Train CycleGAN models.
5. Generate translated images and evaluate KID/FID scores.

## 0. Pre-trained models

We provide pre-trained generators that were used to obtain the `Rethinking
CycleGAN` [paper][uvcgan2_paper] results.
They can be found on [Zenodo][pretrained_models].


## 1. Download Datasets

`uvcgan2` provides a script (`scripts/download_dataset.sh`) to download and
unpack various CycleGAN datasets.

For example, one can use the following commands to download `selfie2anime`,
CelebA `male2female`, CelebA `eyeglasses`, `CelebA-HQ`, and `AFHQ` datasets:

```bash
./scripts/download_dataset.sh selfie2anime
./scripts/download_dataset.sh male2female
./scripts/download_dataset.sh glasses
./scripts/download_dataset.sh celeba_all    # Low-resolution CelebA
./scripts/download_dataset.sh celeba_hq
./scripts/download_dataset.sh afhq
```

The downloaded datasets will be unpacked under the `UVCGAN2_DATA` directory
(or `./data` if `UVCGAN2_DATA` is unset).


## 2. Pre-processing High-Quality Datasets

The images of the high-quality datasets `CelebA-HQ` and `AFHQ` have sizes
of 1024x1024 and 512x512 pixels correspondingly. For the training and
evaluation, however, we have relied on images of size 256x256. The script
`scripts/downsize_right.py` can be used to properly resize the images:

```bash
python3 ./scripts/downsize_right.py -s 256 256 -i lanczos "${UVCGAN2_DATA:-./data}/afhq/"      "${UVCGAN2_DATA:-./data}/afhq_resized_lanczos"
python3 ./scripts/downsize_right.py -s 256 256 -i lanczos "${UVCGAN2_DATA:-./data}/celeba_hq/" "${UVCGAN2_DATA:-./data}/celeba_hq_resized_lanczos"
```

## 3. Generator Pre-training

Once the datasets are ready, the next step is to pre-train generators on the
Inpainting pretext task. `uvcgan2` provides pre-training scripts for all
the datasets:

```
scripts/afhq/pretrain_afhq.py
scripts/anime2selfie/pretrain_anime2selfie.py
scripts/celeba/pretrain_celeba.py
scripts/celeba_hq/pretrain_celebahq.py
```

These scripts can be simply run like
```bash
python3 scripts/afhq/pretrain_afhq.py
```

Optionally, they accept some command line arguments.
For instance, the batch size can be adjusted by:
```bash
python3 scripts/afhq/pretrain_afhq.py --batch-size 8
```

More details can be found by looking over the scripts. Each of them contains
a training configuration, which should be self-explanatory.

When the training is finished, the pre-trained generators will be saved under
the "${UVCGAN2_OUTDIR}" directory.


## 4. Image-to-Image Translation Training


For each of the translation directions, we provide a corresponding image
translation training script:
```
scripts/afhq/train_cat2dog_translation.py
scripts/afhq/train_wild2cat_translation.py
scripts/afhq/train_wild2dog_translation.py
scripts/anime2selfie/train_anime2selfie_translation.py
scripts/celeba/train_celeba_glasses_translation.py
scripts/celeba/train_celeba_male2female_translation.py
scripts/celeba_hq/train_m2f_translation.py
```

Similar to the pre-training scripts, they can be simply run by
```bash
python3 scripts/afhq/train_cat2dog_translation.py
```

The trained models will be saved under the "${UVCGAN_OUTDIR}" directory.


## 5. Evaluation of the trained models

### 5.1 Image Translation

`uvcgan2` provides a script `scripts/translate_images.py` to perform a batch
translation of the images via one of the trained models. The script can
be run as
```bash
python3 scripts/translate_images.py PATH_TO_TRAINED_MODEL --split SPLIT
```

where SPLIT is the split (`train`, `val` or `test`) of the data to translate.
Due to how the datasets are constructed, one should use `test` split for the
`anime2selfie` and `CelebA` datasets, and `val` split for the `CelebA-HQ`
and `AFHQ` datasets.

The translated images will be saved under
`PATH_TO_TRAINED_MODEL/evals/final/images_eval-SPLIT`.


### 5.2 Evaluation of the Quality of Translation

`Rethinking CycleGAN` paper describes two ways to evaluate the quality of
translation:
1. Consistent protocol. Uniform across all datasets.
2. Ad-hoc protocols for `CelebA-HQ` and `AFHQ`.


#### 5.2.1 Consistent Evaluation of the Quality of Translation

The consistent evaluation protocol relies on
[torch_fidelity](https://github.com/toshas/torch-fidelity)
(commit 5f7c5b5ccc4128bd79be2fdd8e75f118aa8fdc7c)
to calculate KID/FID metrics of the translated images.

A helper script `scripts/eval_fid.py` is provided to facilitate such
a calculation. It can be run with
```bash
python3 eval_fid.py `PATH_TO_TRAINED_MODEL/evals/final/images_eval-SPLIT` --kid-size KID_SIZE
```

where `KID_SIZE` is the parameter of the KID calculation algorithm. Its value
depends on the dataset and should be set to match the `Rethinking CycleGAN`
paper. At the end of the calculation, the scores will be saved in the following
file:
```
PATH_TO_TRAINED_MODEL/evals/final/images_eval-SPLIT/fid_metrics.csv
```

Please refer to our Benchmarking [repository][benchmarking_repo] for the
additional details on how the consistent evaluation protocol was applied
to the earlier GAN-based models.


#### 5.2.2 Consistent Evaluation of the Quality of Translation

An alternative way to evaluate `uvcgan2` models is to rely on various
ad-hoc protocols found in the wild. In the paper, we have used two such
protocols for the `CelebA-HQ` and `AFHQ` datasets. For consistency with
previous works, we have used [EGSDE's][egsde_repo] implementation of these
protocols.

The EGSDE's evaluation code can be invoked by running the `run_score.py`
script. The script needs to be manually modified for each translation
direction, but the modifications are straightforward.

An important variable of the `run_score.py` script is `translate_path` that
should be set to point out to the location of the translated images.

Note, however, that the `uvcgan2` changes names of the translated images from
their original, semi-random, values to `sample_1.png`, `sample_2.png`, etc.
The indices correspond to the lexicographically sorted original names.
Before providing the translated images to the `run_score.py` script, they
should be renamed back to the original names.


#### 5.2.3 Consistent Evaluation of the Quality of Translation

Finally, `uvcgan2` provides a script `scripts/eval_il2_scores.py` to batch
evaluate faithfulness scores based on the Inception-v3 L2 distances. Its
invocation is similar to the `scripts/eval_fid.py` from the section 5.2.1.


# F.A.Q.

## I am training my model on a multi-GPU node. How to make sure that I use only one GPU?

You can specify GPUs that `pytorch` will use with the help of the
`CUDA_VISIBLE_DEVICES` environment variable. This variable can be set to a list
of comma-separated GPU indices. When it is set, `pytorch` will only use GPUs
whose IDs are in the `CUDA_VISIBLE_DEVICES`.


# LICENSE

`uvcgan2` is distributed under `BSD-2` license.

`uvcgan2` repository contains some code (primarity in `uvcgan/base`
subdirectory) from [pytorch-CycleGAN-and-pix2pix][cyclegan_repo].
This code is also licensed under `BSD-2` license (please refer to
`uvcgan2/base/LICENSE` for details).

Each code snippet that was taken from
[pytorch-CycleGAN-and-pix2pix][cyclegan_repo] has a note about proper copyright
attribution.

[cyclegan_repo]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
[uvcgan_repo]: https://github.com/LS4GAN/uvcgan
[egsde_repo]: https://github.com/ML-GSAI/EGSDE
[benchmarking_repo]: https://github.com/LS4GAN/benchmarking
[uvcgan2_paper]: https://arxiv.org/abs/2303.16280
[pretrained_models]: https://zenodo.org/record/7813795

