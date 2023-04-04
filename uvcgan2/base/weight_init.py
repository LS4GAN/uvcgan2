# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan2/base/LICENSE` for copyright attribution and LICENSE

import logging
from torch.nn import init

from uvcgan2.torch.select import extract_name_kwargs

LOGGER = logging.getLogger('uvcgan2.base')

def winit_func(m, init_type = 'normal', init_gain = 0.2):
    classname = m.__class__.__name__

    if (
            hasattr(m, 'weight')
        and (classname.find('Conv') != -1 or classname.find('Linear') != -1)
   ):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, init_gain)

        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain = init_gain)

        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')

        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain = init_gain)

        else:
            raise NotImplementedError(
                'Initialization method [%s] is not implemented' % init_type
            )

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, weight_init):
    if weight_init is None:
        return

    name, kwargs = extract_name_kwargs(weight_init)

    LOGGER.debug('Initializnig network with %s', name)
    net.apply(
        lambda m, name=name, kwargs=kwargs : winit_func(m, name, **kwargs)
    )

