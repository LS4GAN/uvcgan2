import torch
from torch import nn

from .select import extract_name_kwargs
from .layers.transformer import calc_tokenized_size

class SequenceRandomMasking(nn.Module):

    def __init__(self, fraction = 0.4, seed = 0, **kwargs):
        super().__init__(**kwargs)
        self._fraction = fraction

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def forward(self, sequence):
        # sequence : (N, L, features)
        mask  = (
              torch.rand((*sequence.shape[:2], 1), generator = self._rng)
            > self._fraction
        )
        return mask.to(sequence.device) * sequence

class ImagePatchRandomMasking(nn.Module):

    def __init__(self, patch_size, fraction = 0.4, seed = 0, **kwargs):
        super().__init__(**kwargs)

        self._patch_size = patch_size
        self._fraction   = fraction

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    def forward(self, image):
        # image : (N, C, H, W)
        N_h, N_w = calc_tokenized_size(image.shape[1:], self._patch_size)

        # mask : (N, 1, N_h, N_w)
        mask = (
              torch.rand((image.shape[0], 1, N_h, N_w), generator = self._rng)
            > self._fraction
        )

        # mask : (N, 1, N_h, N_w)
        #     -> (N, 1,   H,   W)
        mask = mask.repeat_interleave(self._patch_size[0], dim = 2)
        mask = mask.repeat_interleave(self._patch_size[1], dim = 3)

        return mask.to(image.device) * image

# pylint: disable=trailing-whitespace
# class BlockwiseMasking(ImageMaskingBase):
#     # Algorithm 1 of arXiv:2106.08254
# 
#     def __init__(
#         self,
#         mask_ratio     = 0.4,
#         min_block_size = 16,
#         aspect_ratio   = 0.3,
#         seed           = 0,
#     ):
#         self._mask_ratio     = mask_ratio
#         self._min_block_size = min_block_size
#         self._aspect_ratio   = aspect_ratio
# 
#         self._prg = np.random.default_rng(seed)
# 
#     def get_mask_region(self, image, h, w, masking_threshold, masked_patches):
#         min_block_size = self._min_block_size
#         max_block_size = \
#             max(min_block_size, masking_threshold - len(masked_patches))
# 
#         block_size   = self._prg.integers(min_block_size, max_block_size)
#         aspect_ratio = self._prg.uniform(
#             self._aspect_ratio, 1/self._aspect_ratio
#         )
# 
#         y_range = int(np.round(np.sqrt(block_size * aspect_ratio)))
#         x_range = int(np.round(np.sqrt(block_size / aspect_ratio)))
# 
#         y_range = min(y_range, h)
#         x_range = min(x_range, w)
# 
#         y0 = self._prg.integers(0, h - y_range)
#         x0 = self._prg.integers(0, w - x_range)
# 
#         return (y0, x0, y_range, x_range)
# 
#     def mask(self, image):
#         # image : (..., H, W)
#         h = image.shape[-2]
#         w = image.shape[-1]
# 
#         n_patches      = h * w
#         masked_patches = set()
# 
#         masking_threshold = self._mask_ratio * n_patches
# 
#         while len(masked_patches) < masking_threshold:
#             (y0, x0, y_range, x_range) = self.get_mask_region(
#                 image, h, w, masking_threshold, masked_patches
#             )
# 
#             for y in range(y0, y0 + y_range):
#                 for x in range(x0, x0 + x_range):
#                     coord = (x, y)
#                     if coord in masked_patches:
#                         continue
# 
#                     image[..., y, x] = 0
#                     masked_patches.add(coord)
# 
#         return image

def select_masking(masking):
    if masking is None:
        return None

    name, kwargs = extract_name_kwargs(masking)

    if name in [ 'transformer-random', 'sequence-random' ]:
        return SequenceRandomMasking(**kwargs)

    if name == 'image-patch-random':
        return ImagePatchRandomMasking(**kwargs)

    raise ValueError("Unknown masking: '%s'" % name)

