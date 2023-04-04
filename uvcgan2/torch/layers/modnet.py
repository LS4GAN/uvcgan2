# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from uvcgan2.torch.select import get_activ_layer
from .cnn  import get_upsample_x2_layer
from .unet import UNetEncBlock

# Ref: https://arxiv.org/pdf/1912.04958.pdf

def get_demod_scale(mod_scale, weights, eps = 1e-6):
    # Ref: https://arxiv.org/pdf/1912.04958.pdf
    #
    # demod_scale[alpha] = 1 / sqrt(sigma[alpha]^2 + eps)
    #
    # sigma[alpha]^2
    #   = sum_{beta i} (mod_scale[alpha]  * weights[alpha, beta, i])^2
    #   = sum_{beta} (mod_scale[alpha])^2 * sum_i (weights[alpha, beta, i])^2
    #

    # mod_scale : (N, C_in)
    # weights   : (C_out, C_in, h, w)

    # w_sq : (C_out, C_in)
    w_sq = torch.sum(weights.square(), dim = (2, 3))

    # w_sq : (C_out, C_in) -> (1, C_in, C_out)
    w_sq = torch.swapaxes(w_sq, 0, 1).unsqueeze(0)

    # mod_scale_sq : (N, C_in, 1)
    mod_scale_sq = mod_scale.square().unsqueeze(2)

    # sigma : (N, C_out)
    sigma_sq = torch.sum(mod_scale_sq * w_sq, dim = 1)

    # result : (N, C_out)
    return 1 / torch.sqrt(sigma_sq + eps)

class ModulatedConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride = 1, padding = 0, dilation = 1, groups = 1, eps = 1e-6,
        demod = True, device = None, dtype = None, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, device = device, dtype = dtype
        )

        self._eps   = eps
        self._demod = demod

    def forward(self, x, s):
        # x : (N, C_in, H_in, W_in)
        # s : (N, C_in)

        # x_mod : (N, C_in, H_in, W_in)
        x_mod = x * s.unsqueeze(2).unsqueeze(3)

        # y : (N, C_out, H_out, W_out)
        y = self.conv(x_mod)

        if self._demod:
            # s_demod : (N, C_out)
            s_demod = get_demod_scale(s, self.conv.weight)
            y_demod = y * s_demod.unsqueeze(2).unsqueeze(3)

            return y_demod

        return y

class StyleBlock(nn.Module):

    def __init__(
        self, mod_features, style_features, rezero = True, bias = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.affine_mod = nn.Linear(mod_features, style_features, bias = bias)
        self.rezero     = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, mod):
        # mod : (N, mod_features)
        # s   : (N, style_features)
        s = 1 + self.re_alpha * self.affine_mod(mod)

        return s

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class ModNetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, mod_features,
        mid_features = None,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.style_block1 = StyleBlock(
            mod_features, in_features, style_rezero, style_bias
        )
        self.conv_block1 = ModulatedConv2d(
            in_features, mid_features, kernel_size = 3, padding = 1,
            demod = demod
        )
        self.activ1 = get_activ_layer(activ)

        self.style_block2 = StyleBlock(
            mod_features, mid_features, style_rezero, style_bias
        )
        self.conv_block2 = ModulatedConv2d(
            mid_features, out_features, kernel_size = 3, padding = 1,
            demod = demod
        )
        self.activ2 = get_activ_layer(activ)

    def forward(self, x, mod):
        # x   : (N, C_in, H_in, W_in)
        # mod : (N, mod_features)

        # mod_scale1 : (N, C_out)
        mod_scale1 = self.style_block1(mod)

        # y1 : (N, C_mid, H_mid, W_mid)
        y1 = self.conv_block1(x, mod_scale1)
        y1 = self.activ1(y1)

        # mod_scale2 : (N, C_out)
        mod_scale2 = self.style_block2(mod)

        # result : (N, C_out, H_out, W_out)
        result = self.conv_block2(y1, mod_scale2)
        result = self.activ2(result)

        return result

class ModNetDecBlock(nn.Module):

    def __init__(
        self, input_shape, output_features, skip_features, mod_features,
        activ, upsample,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        (input_features, H, W) = input_shape
        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_features
        )

        self.block = ModNetBasicBlock(
            skip_features + input_features, output_features, activ,
            mod_features = mod_features,
            mid_features = max(input_features, input_shape[0]),
            demod        = demod,
            style_rezero = style_rezero,
            style_bias   = style_bias,
        )

        self._output_shape = (output_features, 2 * H, 2 * W)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x, r, mod):
        # x   : (N, C, H_in, W_in)
        # r   : (N, C, H_out, W_out)
        # mod : (N, mod_features)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y, mod)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class ModNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.output_shape
        self.inner_module = None

        self.deconv = ModNetDecBlock(
            input_shape     = self.inner_shape,
            output_features = image_shape[0],
            skip_features   = self.inner_shape[0],
            mod_features    = mod_features,
            activ           = activ,
            upsample        = upsample,
            rezero          = rezero,
            demod           = demod,
            style_rezero    = style_rezero,
            style_bias      = style_bias,
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

        # y   : (N, C_inner, H_inner, W_inner)
        # mod : (N, mod_features)
        y, mod = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r, mod)

        return (y, mod)

class ModNetLinearDecoder(nn.Module):

    def __init__(
        self, features_list, input_shape, output_shape, skip_shapes,
        mod_features, activ, upsample,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
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
            layer = ModNetDecBlock(
                input_shape     = curr_shape,
                output_features = features,
                skip_features   = skip_shape[0],
                mod_features    = mod_features,
                activ           = activ,
                upsample        = upsample,
                rezero          = rezero,
                demod           = demod,
                style_rezero    = style_rezero,
                style_bias      = style_bias,
            )
            curr_shape = layer.output_shape

            self.net.append(layer)

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

    def forward(self, x, skip_list, mod):
        # x   : (N, C, H, W)
        # mod : (N, mod_features)
        # skip_list : List[(N, C_i, H_i, W_i)]

        y = x

        for layer, skip in zip(self.net, skip_list[::-1]):
            y = layer(y, skip, mod)

        return self.output(y)

class ModNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        return_mod   = False,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self.return_mod    = return_mod

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = ModNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                mod_features, rezero, demod, style_rezero, style_bias
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.modnet = unet_layers[0]

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
        result = self.modnet

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
        y, mod = self.modnet(y)
        y = self.layer_output(y)

        if self.return_mod:
            return (y, mod)

        return y

