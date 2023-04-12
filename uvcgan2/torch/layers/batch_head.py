import torch
from torch import nn

from uvcgan2.torch.select import get_activ_layer, extract_name_kwargs

# References:
# arXiv: 1912.0495
# https://github.com/moono/stylegan2-tf-2.x/blob/master/stylegan2/discriminator.py
# https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py

class BatchStdev(nn.Module):

    # pylint: disable=useless-super-delegation
    def __init__(self, **kwargs):
        """ arXiv: 1710.10196 """
        super().__init__(**kwargs)

    @staticmethod
    def safe_stdev(x, dim = 0, eps = 1e-6):
        var   = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        stdev = torch.sqrt(var + eps)

        return stdev

    # pylint: disable=no-self-use
    def forward(self, x):
        """
        NOTE: Reference impl has fixed minibatch size.

        arXiv: 1710.10196

        1. We first compute the standard deviation for each feature in each
           spatial location over the minibatch.

        2. We then average these estimates over all features and spatial
           locations to arrive at a single value.

        3. We replicate the value and concatenate it to all spatial locations
           and over the minibatch, yielding one additional (con-stant) feature
           map.
        """

        # x : (N, C, H, W)
        # x_stdev : (1, C, H, W)
        x_stdev = BatchStdev.safe_stdev(x, dim = 0)

        # x_norm : (1, 1, 1, 1)
        x_norm = torch.mean(x_stdev, dim = (1, 2, 3), keepdim = True)

        # x_norm : (N, 1, H, W)
        x_norm = x_norm.expand((x.shape[0], 1, *x.shape[2:]))

        # y : (N, C + 1, H, W)
        y = torch.cat((x, x_norm), dim = 1)

        return y

class BatchHead1d(nn.Module):

    def __init__(
        self, input_features, mid_features = None, output_features = None,
        activ = 'relu', activ_output = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = input_features

        if output_features is None:
            output_features = mid_features

        self.net = nn.Sequential(
            nn.Linear(input_features, mid_features),
            nn.BatchNorm1d(mid_features),
            get_activ_layer(activ),

            nn.Linear(mid_features, output_features),
            get_activ_layer(activ_output),
        )

    def forward(self, x):
        # x : (N, C)
        return self.net(x)

class BatchHead2d(nn.Module):

    def __init__(
        self, input_features, mid_features = None, output_features = None,
        activ = 'relu', activ_output = None, n_signal = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = input_features

        if output_features is None:
            output_features = mid_features

        self._n_signal = n_signal

        self.norm = nn.BatchNorm2d(input_features)
        self.net  = nn.Sequential(
            nn.Conv2d(
                input_features, mid_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),

            nn.Conv2d(
                mid_features, output_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ_output),
        )

    def forward(self, x):
        # x : (N, C, H, W)
        y = self.norm(x)

        if self._n_signal is not None:
            # Drop queue tokens
            y = y[:self._n_signal, ...]

        return self.net(y)

class BatchStdevHead(nn.Module):

    def __init__(
        self, input_features, mid_features = None, output_features = None,
        activ = 'relu', activ_output = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = input_features

        if output_features is None:
            output_features = mid_features

        self.net = nn.Sequential(
            BatchStdev(),
            nn.Conv2d(
                input_features + 1, mid_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),

            nn.Conv2d(
                mid_features, output_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ_output),
        )

    def forward(self, x):
        # x : (N, C, H, W)
        return self.net(x)

class BatchAverageHead(nn.Module):

    def __init__(
        self, input_features, reduce_channels = True, average_spacial = False,
        activ_output = None, **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        layers = []

        if reduce_channels:
            layers.append(
                nn.Conv2d(input_features, 1, kernel_size = 3, padding = 1)
            )

        if average_spacial:
            layers.append(nn.AdaptiveAvgPool2d(1))

        if activ_output is not None:
            layers.append(get_activ_layer(activ_output))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x : (N, C, H, W)
        return self.net(x)

class BatchHeadWrapper(nn.Module):

    def __init__(self, body, head, **kwargs):
        super().__init__(**kwargs)
        self._body   = body
        self._head   = head

    def forward_head(self, x_body):
        return self._head(x_body)

    def forward_body(self, x):
        return self._body(x)

    def forward(self, x, extra_bodies = None, return_body = False):
        y_body = self._body(x)

        if isinstance(y_body, (list, tuple)):
            y_body_main = list(y_body[:-1])
            y_body_last = y_body[-1]
        else:
            y_body_main = tuple()
            y_body_last = y_body

        if extra_bodies is not None:
            all_bodies = torch.cat((y_body_last, extra_bodies), dim = 0)
            y_head     = self._head(all_bodies)
        else:
            y_head = self._head(y_body_last)

        y_head = y_head[:y_body_last.shape[0]]

        if len(y_body_main) == 0:
            result = y_head
        else:
            result = y_body_main + [ y_head, ]

        if return_body:
            return (result, y_body_last)

        return result

BATCH_HEADS = {
    'batch-norm-1d'  : BatchHead1d,
    'batch-norm-2d'  : BatchHead2d,
    'batch-stdev'    : BatchStdevHead,
    'simple-average' : BatchAverageHead,
    'idt'            : nn.Identity,
}

def get_batch_head(batch_head):
    name, kwargs = extract_name_kwargs(batch_head)

    if name not in BATCH_HEADS:
        raise ValueError("Unknown Batch Head: '%s'" % name)

    return BATCH_HEADS[name](**kwargs)

