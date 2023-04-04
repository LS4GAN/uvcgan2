# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import numpy as np
from torch import nn

from uvcgan2.torch.layers.transformer import (
    calc_tokenized_size, ViTInput, TransformerEncoder, img_to_tokens,
    img_from_tokens
)

class ViTGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape, token_size,
        rescale = False, rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        assert input_shape == output_shape
        image_shape = input_shape

        self.image_shape    = image_shape
        self.token_size     = token_size
        self.token_shape    = (image_shape[0], *token_size)
        self.token_features = np.prod([image_shape[0], *token_size])
        self.N_h, self.N_w  = calc_tokenized_size(image_shape, token_size)
        self.rescale        = rescale

        self.gan_input = ViTInput(
            self.token_features, embed_features, features, self.N_h, self.N_w
        )

        self.trans = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.gan_output = nn.Linear(features, self.token_features)

    # pylint: disable=no-self-use
    def calc_scale(self, x):
        # x : (N, C, H, W)
        return x.abs().mean(dim = (1, 2, 3), keepdim = True) + 1e-8

    def forward(self, x):
        # x : (N, C, H, W)
        if self.rescale:
            scale = self.calc_scale(x)
            x = x / scale

        # itokens : (N, N_h, N_w, C, H_c, W_c)
        itokens = img_to_tokens(x, self.token_shape[1:])

        # itokens : (N, N_h,  N_w, C,  H_c,  W_c)
        #        -> (N, N_h * N_w, C * H_c * W_c)
        #         = (N, L,         in_features)
        itokens = itokens.reshape((itokens.shape[0], self.N_h * self.N_w, -1))

        # y : (N, L, features)
        y = self.gan_input(itokens)
        y = self.trans(y)

        # otokens : (N, L, in_features)
        otokens = self.gan_output(y)

        # otokens : (N, L, in_features)
        #        -> (N, N_h, N_w, C, H_c, W_c)
        otokens = otokens.reshape((
            otokens.shape[0], self.N_h, self.N_w, *self.token_shape
        ))

        result = img_from_tokens(otokens)
        if self.rescale:
            result = result * scale

        return result

