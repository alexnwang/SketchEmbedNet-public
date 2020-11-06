from models import register_model, VAE
from models.subs.conv_block import BlockConv, BlockConvReverse

import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except:
    hvd = None


@register_model('vae_enc_block')
class VAEEncBlock(VAE):
    def __init__(self, base_dir, model_id, params, training=True):
        """
        Extension of the VAE using a Conv4 backbone
        :param base_dir:
        :param model_id:
        :param params:
        :param training:
        """
        # ----- Init Model ----- #
        super(VAEEncBlock, self).__init__(base_dir, model_id, params, training)

    def _build_model(self):
        self._conv_encoder = tf.keras.Sequential([
            BlockConv(64),
            BlockConv(64),
            BlockConv(64),
            BlockConv(64),
            tf.keras.layers.GlobalAveragePooling2D()
        ])

        self._mu = tf.keras.layers.Dense(self._latent_size)
        self._var = tf.keras.layers.Dense(self._latent_size)

        self._conv_decoder = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D((self._png_dim//16, self._png_dim//16)),
            BlockConvReverse(64),
            BlockConvReverse(64),
            BlockConvReverse(64),
            BlockConvReverse(1 if self._grayscale else 3)
        ])

        lr = tf.keras.optimizers.schedules.ExponentialDecay(self._lr, self._lr_decay_step, self._lr_decay_rate)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr)
