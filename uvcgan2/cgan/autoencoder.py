# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.encoder is not callable (not-callable)
from uvcgan2.torch.select             import select_optimizer, select_loss
from uvcgan2.torch.image_masking      import select_masking
from uvcgan2.models.generator         import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input

class Autoencoder(ModelBase):

    def _setup_images(self, _config):
        images = [ 'real_a', 'reco_a', 'real_b', 'reco_b', ]

        if self.masking is not None:
            images += [ 'masked_a', 'masked_b' ]

        return NamedDict(*images)

    def _setup_models(self, config):
        if self.joint:
            image_shape = config.data.datasets[0].shape

            assert image_shape == config.data.datasets[1].shape, (
                "Joint autoencoder requires all datasets to have "
                "the same image shape"
            )

            return NamedDict(
                encoder = construct_generator(
                    config.generator, image_shape, image_shape, self.device
                )
            )

        models = NamedDict('encoder_a', 'encoder_b')
        models.encoder_a = construct_generator(
            config.generator,
            config.data.datasets[0].shape,
            config.data.datasets[0].shape,
            self.device
        )
        models.encoder_b = construct_generator(
            config.generator,
            config.data.datasets[1].shape,
            config.data.datasets[1].shape,
            self.device
        )

        return models

    def _setup_losses(self, config):
        self.loss_fn = select_loss(config.loss)

        assert config.gradient_penalty is None, \
            "Autoencoder model does not support gradient penalty"

        return NamedDict('loss_a', 'loss_b')

    def _setup_optimizers(self, config):
        if self.joint:
            return NamedDict(
                encoder = select_optimizer(
                    self.models.encoder.parameters(),
                    config.generator.optimizer
                )
            )

        optimizers = NamedDict('encoder_a', 'encoder_b')

        optimizers.encoder_a = select_optimizer(
            self.models.encoder_a.parameters(), config.generator.optimizer
        )
        optimizers.encoder_b = select_optimizer(
            self.models.encoder_b.parameters(), config.generator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device,
        joint = False, masking = None
    ):
        # pylint: disable=too-many-arguments
        self.joint   = joint
        self.masking = select_masking(masking)

        assert len(config.data.datasets) == 2, \
            "Autoencoder expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        assert config.discriminator is None, \
            "Autoencoder model does not use discriminator"

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

    def forward(self):
        input_a = self.images.real_a
        input_b = self.images.real_b

        if self.masking is not None:
            if input_a is not None:
                input_a = self.masking(input_a)

            if input_b is not None:
                input_b = self.masking(input_b)

            self.images.masked_a = input_a
            self.images.masked_b = input_b

        if input_a is not None:
            if self.joint:
                self.images.reco_a = self.models.encoder  (input_a)
            else:
                self.images.reco_a = self.models.encoder_a(input_a)

        if input_b is not None:
            if self.joint:
                self.images.reco_b = self.models.encoder  (input_b)
            else:
                self.images.reco_b = self.models.encoder_b(input_b)

    def backward_generator_base(self, real, reco):
        loss = self.loss_fn(reco, real)
        loss.backward()

        return loss

    def backward_generators(self):
        self.losses.loss_b = self.backward_generator_base(
            self.images.real_b, self.images.reco_b
        )

        self.losses.loss_a = self.backward_generator_base(
            self.images.real_a, self.images.reco_a
        )

    def optimization_step(self):
        self.forward()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        self.backward_generators()

        for optimizer in self.optimizers.values():
            optimizer.step()

