# LICENSE
# This file was extracted from
#   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Please see `uvcgan2/base/LICENSE` for copyright attribution and LICENSE

# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import torch

from uvcgan2.torch.select         import select_optimizer
from uvcgan2.base.losses          import GANLoss, cal_gradient_penalty
from uvcgan2.models.discriminator import construct_discriminator
from uvcgan2.models.generator     import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input

class Pix2PixModel(ModelBase):

    def _setup_images(self, _config):
        return NamedDict('real_a', 'fake_b', 'real_b', 'fake_a')

    def _setup_models(self, config):
        models = { }

        image_shape_a = config.data.datasets[0].shape
        image_shape_b = config.data.datasets[1].shape

        assert image_shape_a[1:] == image_shape_b[1:], \
            "Pix2Pix needs images in both domains to have the same size"

        models['gen_ab'] = construct_generator(
            config.generator, image_shape_a, image_shape_b, self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator, image_shape_b, image_shape_a, self.device
        )

        if self.is_train:
            extended_image_shape = (
                image_shape_a[0] + image_shape_b[0], *image_shape_a[1:]
            )

            for name in [ 'disc_a', 'disc_b' ]:
                models[name] = construct_discriminator(
                    config.discriminator, extended_image_shape, self.device
                )

        return NamedDict(**models)

    def _setup_losses(self, config):
        return NamedDict(
            'gen_ab', 'gen_ba', 'l1_ab', 'l1_ba', 'disc_a', 'disc_b'
        )

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen_ab', 'gen_ba', 'disc_a', 'disc_b')

        optimizers.gen_ab = select_optimizer(
            self.models.gen_ab.parameters(), config.generator.optimizer
        )
        optimizers.gen_ba = select_optimizer(
            self.models.gen_ba.parameters(), config.generator.optimizer
        )

        optimizers.disc_a = select_optimizer(
            self.models.disc_a.parameters(), config.discriminator.optimizer
        )
        optimizers.disc_b = select_optimizer(
            self.models.disc_b.parameters(), config.discriminator.optimizer
        )

        return optimizers

    def __init__(self, savedir, config, is_train, device):
        super().__init__(savedir, config, is_train, device)

        assert len(config.data.datasets) == 2, \
            "Pix2Pix expects a pair of datasets"

        self.criterion_gan    = GANLoss(config.loss).to(self.device)
        self.criterion_l1     = torch.nn.L1Loss()
        self.gradient_penalty = config.gradient_penalty

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

    def forward(self):
        if self.images.real_a is not None:
            self.images.fake_b = self.models.gen_ab(self.images.real_a)

        if self.images.real_b is not None:
            self.images.fake_a = self.models.gen_ba(self.images.real_b)

    def backward_discriminator_base(self, model, real, fake, preimage):
        cond_real = torch.cat([real, preimage], dim = 1)
        cond_fake = torch.cat([fake, preimage], dim = 1).detach()

        pred_real = model(cond_real)
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = model(cond_fake)
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5

        if self.gradient_penalty is not None:
            loss += cal_gradient_penalty(
                model, cond_real, cond_fake, real.device,
                **self.gradient_penalty
            )[0]

        loss.backward()
        return loss

    def backward_discriminators(self):
        self.losses.disc_b = self.backward_discriminator_base(
            self.models.disc_b,
            self.images.real_b, self.images.fake_b, self.images.real_a
        )

        self.losses.disc_a = self.backward_discriminator_base(
            self.models.disc_a,
            self.images.real_a, self.images.fake_a, self.images.real_b
        )

    def backward_generator_base(self, disc, real, fake, preimage):
        loss_gen = self.criterion_gan(
            disc(torch.cat([fake, preimage], dim = 1)), True
        )

        loss_l1 = self.criterion_l1(fake, real)

        loss = loss_gen + loss_l1
        loss.backward()

        return (loss_gen, loss_l1)

    def backward_generators(self):
        self.losses.gen_ab, self.losses.l1_ab = self.backward_generator_base(
            self.models.disc_b,
            self.images.real_b, self.images.fake_b, self.images.real_a
        )

        self.losses.gen_ba, self.losses.l1_ba = self.backward_generator_base(
            self.models.disc_a,
            self.images.real_a, self.images.fake_a, self.images.real_b
        )

    def optimization_step(self):
        self.forward()

        # Generators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        self.optimizers.gen_ab.zero_grad()
        self.optimizers.gen_ba.zero_grad()
        self.backward_generators()
        self.optimizers.gen_ab.step()
        self.optimizers.gen_ba.step()

        # Discriminators
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        self.optimizers.disc_a.zero_grad()
        self.optimizers.disc_b.zero_grad()
        self.backward_discriminators()
        self.optimizers.disc_a.step()
        self.optimizers.disc_b.step()

