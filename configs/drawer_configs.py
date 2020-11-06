from util import HParams, ST1_classes
from configs.base import register_config

@register_config("drawer")
def drawer_default():
    return HParams(
        # ----- Model Parameters ----- #
        rnn_cell="hyper",
        rnn_output_size=1024,
        z_size=256,
        num_mixture=30,

        kl_tolerance=0.2,
        kl_weight=0.0,

        pixel_loss_weight_max=1.0,
        pixel_loss_weight_min=0.0,
        pixel_loss_weight_interval=0.0,
        pixel_loss_step=0,

        sigma_decay_start=200000.0,
        sigma_decay_rate=1.0,
        sigma_decay_freq=10000,
        sigma_init=2.0,

        cell_configs={"hyper_num_units": 512, "hyper_embedding_size": 64, "use_recurrent_dropout": False, "recurrent_dropout_prob": 0.9},

        # ----- Model Specific Training Parameters ----- #
        lr=0.001,
        lr_decay_freq=15000,
        lr_decay_rate=0.85,

        gradient_cap=1.0,

        # ----- Other Configs ----- #
        distributed=False,
    )