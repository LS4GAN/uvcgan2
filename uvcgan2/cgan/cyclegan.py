# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan2/base/LICENSE` for copyright attribution and LICENSE

# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch

from uvcgan2.torch.select         import select_optimizer
from uvcgan2.base.image_pool      import ImagePool
from uvcgan2.base.losses          import GANLoss, cal_gradient_penalty
from uvcgan2.models.discriminator import construct_discriminator
from uvcgan2.models.generator     import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input

class CycleGANModel(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [ 'real_a', 'fake_b', 'reco_a', 'real_b', 'fake_a', 'reco_b' ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b' ]

        return NamedDict(*images)

    def _setup_models(self, config):
        models = {}

        models['gen_ab'] = construct_generator(
            config.generator,
            config.data.datasets[0].shape,
            config.data.datasets[1].shape,
            self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator,
            config.data.datasets[1].shape,
            config.data.datasets[0].shape,
            self.device
        )

        if self.is_train:
            models['disc_a'] = construct_discriminator(
                config.discriminator,
                config.data.datasets[0].shape,
                self.device
            )
            models['disc_b'] = construct_discriminator(
                config.discriminator,
                config.data.datasets[1].shape,
                self.device
            )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'cycle_a', 'cycle_b', 'disc_a', 'disc_b'
        ]

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, pool_size = 50,
        lambda_a = 10.0, lambda_b = 10.0, lambda_idt = 0.5
    ):
        # pylint: disable=too-many-arguments
        self.lambda_a   = lambda_a
        self.lambda_b   = lambda_b
        self.lambda_idt = lambda_idt

        assert len(config.data.datasets) == 2, \
            "CycleGAN expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan    = GANLoss(config.loss).to(self.device)
        self.gradient_penalty = config.gradient_penalty
        self.criterion_cycle  = torch.nn.L1Loss()
        self.criterion_idt    = torch.nn.L1Loss()

        if self.is_train:
            self.pred_a_pool = ImagePool(pool_size)
            self.pred_b_pool = ImagePool(pool_size)

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

    def forward(self):
        def simple_fwd(batch, gen_fwd, gen_bkw):
            if batch is None:
                return (None, None)

            fake = gen_fwd(batch)
            reco = gen_bkw(fake)

            return (fake, reco)

        self.images.fake_b, self.images.reco_a = simple_fwd(
            self.images.real_a, self.models.gen_ab, self.models.gen_ba
        )

        self.images.fake_a, self.images.reco_b = simple_fwd(
            self.images.real_b, self.models.gen_ba, self.models.gen_ab
        )

    def backward_discriminator_base(self, model, real, fake):
        pred_real = model(real)
        loss_real = self.criterion_gan(pred_real, True)

        #
        # NOTE:
        #   This is a workaround to a pytorch 1.9.0 bug that manifests when
        #   cudnn is enabled. When the bug is solved remove no_grad block and
        #   replace `model(fake)` by `model(fake.detach())`.
        #
        #   bug: https://github.com/pytorch/pytorch/issues/48439
        #
        with torch.no_grad():
            fake = fake.contiguous()

        pred_fake = model(fake)
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5

        if self.gradient_penalty is not None:
            loss += cal_gradient_penalty(
                model, real, fake, real.device, **self.gradient_penalty
            )[0]

        loss.backward()
        return loss

    def backward_discriminators(self):
        fake_a = self.pred_a_pool.query(self.images.fake_a)
        fake_b = self.pred_b_pool.query(self.images.fake_b)

        self.losses.disc_b = self.backward_discriminator_base(
            self.models.disc_b, self.images.real_b, fake_b
        )

        self.losses.disc_a = self.backward_discriminator_base(
            self.models.disc_a, self.images.real_a, fake_a
        )

    def backward_generators(self):
        lambda_idt = self.lambda_idt
        lambda_a   = self.lambda_a
        lambda_b   = self.lambda_b

        self.losses.gen_ab = self.criterion_gan(
            self.models.disc_b(self.images.fake_b), True
        )
        self.losses.gen_ba = self.criterion_gan(
            self.models.disc_a(self.images.fake_a), True
        )
        self.losses.cycle_a = lambda_a * self.criterion_cycle(
            self.images.reco_a, self.images.real_a
        )
        self.losses.cycle_b = lambda_b * self.criterion_cycle(
            self.images.reco_b, self.images.real_b
        )

        loss = (
              self.losses.gen_ab  + self.losses.gen_ba
            + self.losses.cycle_a + self.losses.cycle_b
        )

        if lambda_idt > 0:
            self.images.idt_b = self.models.gen_ab(self.images.real_b)
            self.losses.idt_b = lambda_b * lambda_idt * self.criterion_idt(
                self.images.idt_b, self.images.real_b
            )

            self.images.idt_a = self.models.gen_ba(self.images.real_a)
            self.losses.idt_a = lambda_a * lambda_idt * self.criterion_idt(
                self.images.idt_a, self.images.real_a
            )

            loss += (self.losses.idt_a + self.losses.idt_b)

        loss.backward()

    def optimization_step(self):
        self.forward()

        # Generators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen.zero_grad()
        self.backward_generators()
        self.optimizers.gen.step()

        # Discriminators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc.zero_grad()
        self.backward_discriminators()
        self.optimizers.disc.step()

