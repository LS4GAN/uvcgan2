# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from uvcgan2.torch.select import get_norm_layer, get_activ_layer

from .cnn import get_downsample_x2_layer, get_upsample_x2_layer

class UnetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
            get_activ_layer(activ),

            get_norm_layer(norm, mid_features),
            nn.Conv2d(
                mid_features, out_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x)

class UNetEncBlock(nn.Module):

    def __init__(
        self, features, activ, norm, downsample, input_shape, **kwargs
    ):
        super().__init__(**kwargs)

        self.downsample, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = UnetBasicBlock(C, features, activ, norm)

        self._output_shape = (output_features, H//2, W//2)

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return (y, r)

class UNetDecBlock(nn.Module):

    def __init__(
        self, input_shape, output_features, skip_features, activ, norm,
        upsample, rezero = False, **kwargs
    ):
        super().__init__(**kwargs)

        (input_features, H, W) = input_shape
        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_features
        )

        self.block = UnetBasicBlock(
            input_features + skip_features, output_features, activ, norm,
            mid_features = max(skip_features, input_features, output_features)
        )

        self._output_shape = (output_features, 2 * H, 2 * W)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x, r):
        # x : (N, C, H_in, W_in)
        # r : (N, C_skip, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C_skip + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class UNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.output_shape
        self.inner_module = None

        self.deconv = UNetDecBlock(
            self.inner_shape, image_shape[0], self.inner_shape[0],
            activ, norm, upsample, rezero
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y : (N, C_inner, H_inner, W_inner)
        y = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r)

        return y

class UNetLinearEncoder(nn.Module):

    def __init__(
        self, features_list, image_shape, activ, norm, downsample, **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self._skip_shapes  = []

        self.net   = nn.ModuleList()
        curr_shape = image_shape

        for features in features_list:
            layer = UNetEncBlock(features, activ, norm, downsample, curr_shape)
            self.net.append(layer)

            curr_shape = layer.output_shape
            self._skip_shapes.append(curr_shape)

        self._output_shape = curr_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def skip_shapes(self):
        return self._skip_shapes

    def forward(self, x, return_skips = True):
        # x : (N, C, H, W)

        skips = [ ]
        y = x

        for layer in self.net:
            y, r = layer(y)
            if return_skips:
                skips.append(r)

        if return_skips:
            return y, skips
        else:
            return y

class UNetLinearDecoder(nn.Module):

    def __init__(
        self, features_list, input_shape, output_shape, skip_shapes,
        activ, norm, upsample, **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.net           = nn.ModuleList()
        self._input_shape  = input_shape
        self._output_shape = output_shape
        curr_shape         = input_shape

        for features, skip_shape in zip(
            features_list[::-1], skip_shapes[::-1]
        ):
            layer = UNetDecBlock(
                curr_shape, features, skip_shape[0], activ, norm, upsample
            )
            curr_shape = layer.output_shape

            self.net.append(layer)

        if output_shape[0] == curr_shape[0]:
            self.output = nn.Identity()
        else:
            self.output = nn.Conv2d(
                curr_shape[0], output_shape[0], kernel_size = 1
            )

        curr_shape = (output_shape[0], *curr_shape[1:])
        assert tuple(output_shape) == tuple(curr_shape)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x, skip_list):
        # x : (N, C, H, W)
        # skip_list : List[(N, C_i, H_i, W_i)]

        y = x

        for layer, skip in zip(self.net, skip_list[::-1]):
            y = layer(y, skip)

        return self.output(y)

class UNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        rezero = True, **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = UNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                rezero
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.unet = unet_layers[0]

    def _construct_input_layer(self, activ):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self):
        self.layer_output = nn.Conv2d(
            self.features_list[0], self.image_shape[0], kernel_size = 1
        )

    def get_innermost_block(self):
        result = self.unet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W)

        y = self.layer_input(x)
        y = self.unet(y)
        y = self.layer_output(y)

        return y

