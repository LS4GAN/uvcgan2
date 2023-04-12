# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from uvcgan2.torch.select import get_norm_layer, get_activ_layer

def calc_tokenized_size(image_shape, token_size):
    # image_shape : (C, H, W)
    # token_size  : (H_t, W_t)
    if image_shape[1] % token_size[0] != 0:
        raise ValueError(
            "Token width %d does not divide image width %d" % (
                token_size[0], image_shape[1]
            )
        )

    if image_shape[2] % token_size[1] != 0:
        raise ValueError(
            "Token height %d does not divide image height %d" % (
                token_size[1], image_shape[2]
            )
        )

    # result : (N_h, N_w)
    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1])

def img_to_tokens(image_batch, token_size):
    # image_batch : (N, C, H, W)
    # token_size  : (H_t, W_t)

    # result : (N, C, N_h, H_t, W)
    result = image_batch.view(
        (*image_batch.shape[:2], -1, token_size[0], image_batch.shape[3])
    )

    # result : (N, C, N_h, H_t, W       )
    #       -> (N, C, N_h, H_t, N_w, W_t)
    result = result.view((*result.shape[:4], -1, token_size[1]))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, N_h, N_w, C, H_t, W_t)
    result = result.permute((0, 2, 4, 1, 3, 5))

    return result

def img_from_tokens(tokens):
    # tokens : (N, N_h, N_w, C, H_t, W_t)
    # result : (N, C, N_h, H_t, N_w, W_t)
    result = tokens.permute((0, 3, 1, 4, 2, 5))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, C, N_h, H_t, N_w * W_t)
    #        = (N, C, N_h, H_t, W)
    result = result.reshape((*result.shape[:4], -1))

    # result : (N, C, N_h, H_t, W)
    #       -> (N, C, N_h * H_t, W)
    #        = (N, C, H, W)
    result = result.reshape((*result.shape[:2], -1, result.shape[4]))

    return result

def img_to_pixelwise_tokens(image):
    # image : (N, C, H, W)

    # result : (N, C, H * W)
    result = image.view(*image.shape[:2], -1)

    # result : (N, C,     H * W)
    #       -> (N, H * W, C    )
    #        = (N, L,     C)
    result = result.permute((0, 2, 1))

    # (N, L, C)
    return result

def img_from_pixelwise_tokens(tokens, image_shape):
    # tokens      : (N, L, C)
    # image_shape : (3, )

    # tokens : (N, L, C)
    #       -> (N, C, L)
    #        = (N, C, H * W)
    tokens = tokens.permute((0, 2, 1))

    # (N, C, H, W)
    return tokens.view(*tokens.shape[:2], *image_shape[1:])

class PositionWiseFFN(nn.Module):

    def __init__(self, features, ffn_features, activ = 'gelu', **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(features, ffn_features),
            get_activ_layer(activ),
            nn.Linear(ffn_features, features),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, activ = 'gelu', norm = None,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = get_norm_layer(norm, features)
        self.atten = nn.MultiheadAttention(features, n_heads)

        self.norm2 = get_norm_layer(norm, features)
        self.ffn   = PositionWiseFFN(features, ffn_features, activ)

        self.rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x: (L, N, features)

        # Step 1: Multi-Head Self Attention
        y1 = self.norm1(x)
        y1, _atten_weights = self.atten(y1, y1, y1)

        y  = x + self.re_alpha * y1

        # Step 2: PositionWise Feed Forward Network
        y2 = self.norm2(y)
        y2 = self.ffn(y2)

        y  = y + self.re_alpha * y2

        return y

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class TransformerEncoder(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, n_blocks, activ, norm,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(*[
            TransformerBlock(
                features, ffn_features, n_heads, activ, norm, rezero
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x : (N, L, features)

        # y : (L, N, features)
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result

class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, features, height, width, **kwargs):
        super().__init__(**kwargs)
        self.projector = nn.Linear(2, features)
        self._height   = height
        self._width    = width

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        x_norm = 2 * x / (self._width  - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2)
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim = 2)

        return torch.sin(self.projector(z))

class ViTInput(nn.Module):

    def __init__(
        self, input_features, embed_features, features, height, width,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._height   = height
        self._width    = width

        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)

        x, y   = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed  = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # x     : (N, L, input_features)
        # embed : (1, height * width, embed_features)
        #       = (1, L, embed_features)
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1, L, embed_features)
        #      -> (N, L, embed_features)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features)
        result = torch.cat([embed, x], dim = 2)

        # (N, L, features)
        return self.output(result)

class PixelwiseViT(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, C, H * W)
        itokens = x.view(*x.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        itokens = itokens.permute((0, 2, 1))

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W)
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W)
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        return result

class ExtendedTransformerEncoder(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, activ, norm,
        rezero = True, n_ext = 1, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.extra_tokens = nn.Parameter(torch.empty((1, n_ext, features)))
        torch.nn.init.normal_(self.extra_tokens)

    def forward(self, x):
        # x : (N, L, C)
        N, L, _C = x.shape

        # i_extra_tokens : (N, n_extra, C)
        i_extra_tokens = self.extra_tokens.tile(N, 1, 1)

        # y : (N, L + n_extra, C)
        y = torch.cat([ x, i_extra_tokens ], dim = 1)
        y = self.encoder(y)

        # o_extra_tokens : (N, n_extra, features)
        o_extra_tokens = y[:, L:, :]

        # result : (N, L, C)
        result = y[:, :L, :]

        return (result, o_extra_tokens.reshape(N, -1))

class ExtendedPixelwiseViT(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, rezero = True, n_ext = 1, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.extra_tokens = nn.Parameter(torch.empty((1, n_ext, features)))
        torch.nn.init.normal_(self.extra_tokens)

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, L, C)
        itokens    = img_to_pixelwise_tokens(x)
        (N, L, _C) = itokens.shape

        # i_extra_tokens : (N, n_extra, C)
        i_extra_tokens = self.extra_tokens.tile(itokens.shape[0], 1, 1)

        # y : (N, L, features)
        y = self.trans_input(itokens)

        # y : (N, L + n_extra, C)
        y = torch.cat([ y, i_extra_tokens ], dim = 1)
        y = self.encoder(y)

        # o_extra_tokens : (N, n_extra, features)
        o_extra_tokens = y[:, L:, :]

        # otokens : (N, L, C)
        otokens = self.trans_output(y[:, :L, :])

        # result : (N, C, H, W)
        result = img_from_pixelwise_tokens(otokens, self.image_shape)

        return (result, o_extra_tokens.reshape(N, -1))

