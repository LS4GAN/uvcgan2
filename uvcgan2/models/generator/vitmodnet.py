# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from torch import nn

from uvcgan2.torch.layers.transformer import ExtendedPixelwiseViT
from uvcgan2.torch.layers.modnet      import ModNet
from uvcgan2.torch.select             import get_activ_layer

class ViTModNetGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, input_shape, output_shape, modnet_features_list,
        modnet_activ,
        modnet_norm       = None,
        modnet_downsample = 'conv',
        modnet_upsample   = 'upsample-conv',
        modnet_rezero     = False,
        modnet_demod      = True,
        rezero            = True,
        activ_output      = None,
        style_rezero      = True,
        style_bias        = True,
        n_ext             = 1,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        assert input_shape == output_shape
        image_shape = input_shape

        self.image_shape = image_shape

        mod_features = features * n_ext

        self.net = ModNet(
            modnet_features_list, modnet_activ, modnet_norm, image_shape,
            modnet_downsample, modnet_upsample, mod_features, modnet_rezero,
            modnet_demod, style_rezero, style_bias, return_mod = False
        )

        bottleneck = ExtendedPixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = self.net.get_inner_shape(),
            rezero      = rezero,
            n_ext       = n_ext,
        )

        self.net.set_bottleneck(bottleneck)

        self.output = get_activ_layer(activ_output)

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)
        return self.output(result)

