from models import DrawerModel, register_model
from models.subs.decoders import DecoderDefault
from models.subs.encoders import EncoderResnet12

import tensorflow as tf
try:
    import horovod.tensorflow as hvd
except:
    hvd = None


@register_model('drawer_enc_resnet12')
class DrawerEncResnet12Model(DrawerModel):
    def __init__(self, base_dir, model_id, params, training=True, ckpt=None):
        """
        Inherit from the base Drawer but using the Conv4 encoder backbone from https://arxiv.org/pdf/1709.04121.pdf
        :param base_dir:
        :param model_id:
        :param params:
        :param training:
        :param ckpt:
        """
        # ----- Init Model ----- #
        super(DrawerEncResnet12Model, self).__init__(base_dir, model_id, params, training, ckpt=ckpt)

    def _build_model(self):
        self._decoder = DecoderDefault(self._dec_rnn_size, self._num_mixture, self._rnn_model, self._cell_configs)
        self._encoder = EncoderResnet12(self._z_size, self._decoder.cell.state_size)

        lr_init = self._lr if not self._distributed else self._lr * hvd.size()
        lr = tf.keras.optimizers.schedules.ExponentialDecay(lr_init, self._lr_decay_freq, self._lr_decay_rate)
        self._optimizer = tf.optimizers.Adam(learning_rate=lr, clipvalue=self._gradient_cap)
