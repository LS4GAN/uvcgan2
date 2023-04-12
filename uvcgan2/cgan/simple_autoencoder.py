from uvcgan2.torch.select             import select_optimizer, select_loss
from uvcgan2.torch.image_masking      import select_masking
from uvcgan2.models.generator         import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict

class SimpleAutoencoder(ModelBase):
    """Model that tries to train an autoencoder (i.e. target == input).

    This autoencoder expects inputs to be either tuples of the form
    `(features, target)` or the `features` itself.
    """

    def _setup_images(self, _config):
        images = [ 'real', 'reco' ]

        if self.masking is not None:
            images.append('masked')

        return NamedDict(*images)

    def _setup_models(self, config):
        return NamedDict(
            encoder = construct_generator(
                config.generator,
                config.data.datasets[0].shape,
                config.data.datasets[0].shape,
                self.device
            )
        )

    def _setup_losses(self, config):
        self.loss_fn = select_loss(config.loss)

        assert config.gradient_penalty is None, \
            "Autoencoder model does not support gradient penalty"

        return NamedDict('loss')

    def _setup_optimizers(self, config):
        return NamedDict(
            encoder = select_optimizer(
                self.models.encoder.parameters(), config.generator.optimizer
            )
        )

    def __init__(
        self, savedir, config, is_train, device, masking = None
    ):
        # pylint: disable=too-many-arguments
        self.masking = select_masking(masking)
        assert len(config.data.datasets) == 1, \
            "Simple Autoencoder can work only with a single dataset"

        super().__init__(savedir, config, is_train, device)

        assert config.discriminator is None, \
            "Autoencoder model does not use discriminator"

    def _set_input(self, inputs, _domain):
        # inputs : image or (image, label)
        if isinstance(inputs, (list, tuple)):
            self.images.real = inputs[0].to(self.device)
        else:
            self.images.real = inputs.to(self.device)

    def forward(self):
        if self.masking is None:
            input_img = self.images.real
        else:
            self.images.masked = self.masking(self.images.real)
            input_img          = self.images.masked

        self.images.reco = self.models.encoder(input_img)

    def backward(self):
        loss = self.loss_fn(self.images.reco, self.images.real)
        loss.backward()

        self.losses.loss = loss

    def optimization_step(self):
        self.forward()

        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        self.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()

