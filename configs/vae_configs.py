from util import HParams
from configs.base import register_config


@register_config("vae")
def vae_default():
    return HParams(
        # ----- Model Parameters ----- #
        latent_size=256,
        png_dim=32,
        grayscale=False,

        kl_weight=1.0,
        kl_tolerance=0.0,

        filters=[64, 128, 256, 512],

        # ----- Model Specific Training Parameters ----- #
        lr=0.001,
        lr_decay_step=15000,
        lr_decay_rate=0.85,
    )

@register_config("vae/natural")
def vae_natural(hparams: HParams):
    hparams.set_hparam("png_dim", 96)

    return hparams

@register_config("vae/ae")
def vae_ae(hparams: HParams):
    hparams.set_hparam("png_dim", 96)

    return hparams

@register_config("vae/grayscale")
def vae_grayscale(hparams: HParams):
    hparams.set_hparam("grayscale", True)
    return hparams