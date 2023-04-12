import torch

def mix_tensors(alpha, a, b):
    return alpha * a + (1 - alpha) * b

def recursively_mix_args(alpha, a, b):
    if a is None:
        assert b is None
        return None

    if isinstance(a, (tuple, list)):
        return type(a)(
            recursively_mix_args(alpha, x, y) for (x, y) in zip(a, b)
        )

    if isinstance(a, dict):
        return {
            k : recursively_mix_args(alpha, a[k], b[k]) for k in a
        }

    if isinstance(a, torch.Tensor):
        return mix_tensors(alpha, a, b)

    assert a == b
    return a

def reduce_tensor(y, reduction, reduce_batch = False):
    if isinstance(y, (list, tuple)):
        return [ reduce_tensor(x, reduction) for x in y ]

    if (reduction is None) or (reduction == 'none'):
        return y

    if reduction == 'mean':
        if reduce_batch:
            return y.mean()
        else:
            return y.mean(dim = tuple(i for i in range(1, y.dim())))

    if reduction == 'sum':
        if reduce_batch:
            return y.sum()
        else:
            return y.sum(dim = tuple(i for i in range(1, y.dim())))

    raise ValueError(f"Unknown reduction: '{reduction}'")

class GradientPenalty:

    def __init__(
        self, mix_type, center, lambda_gp, seed = 0,
        reduction = 'mean', gp_reduction = 'mean'
    ):
        # pylint: disable=too-many-arguments
        self._mix_type  = mix_type
        self._center    = center
        self._reduce    = reduction
        self._gp_reduce = gp_reduction
        self._lambda    = lambda_gp

        self._rng      = torch.Generator()
        self._rng.manual_seed(seed)

    def eval_at(self, model, x, **model_kwargs):
        x.requires_grad_(True)

        y = model(x, **model_kwargs)
        y = reduce_tensor(y, self._reduce, reduce_batch = False)

        if not isinstance(y, list):
            y = [ y, ]

        grad = torch.autograd.grad(
            outputs      = y,
            inputs       = x,
            grad_outputs = [ torch.ones(z.size()).to(z.device) for z in y ],
            create_graph = True,
            retain_graph = True,
        )

        grad_x = grad[0].reshape((x.shape[0], -1))
        result = torch.square(
            torch.norm(grad_x, p = 2, dim = 1) - self._center
        )

        return self._lambda * result

    def mix_gp_args(self, a, b, model_kwargs_a, model_kwargs_b):
        alpha  = torch.rand(1, generator = self._rng).to(a.device)
        result = mix_tensors(alpha, a, b)

        mixed_kwargs = recursively_mix_args(
            alpha, model_kwargs_a, model_kwargs_b
        )
        return (result, mixed_kwargs)

    def get_eval_point(
        self, fake, real, model_kwargs_fake = None, model_kwargs_real = None
    ):
        if self._mix_type == 'real':
            return (real.clone(), model_kwargs_real)

        if self._mix_type == 'fake':
            return (fake.clone(), model_kwargs_fake)

        if self._mix_type == 'real-fake':
            return self.mix_gp_args(
                real, fake, model_kwargs_real, model_kwargs_fake
            )

        raise ValueError(f"Unknown mix type: {self._mix_type}")

    def __call__(
        self, model, fake, real,
        model_kwargs_fake = None,
        model_kwargs_real = None,
    ):
        # pylint: disable=too-many-arguments
        x, model_kwargs = self.get_eval_point(
            fake, real, model_kwargs_fake, model_kwargs_real
        )

        model_kwargs = model_kwargs or {}
        result       = self.eval_at(model, x, **model_kwargs)

        return reduce_tensor(result, self._gp_reduce, reduce_batch = True)

