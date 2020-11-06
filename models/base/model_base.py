import os

import tensorflow as tf


class BaseModel(object):
    """
    Basic model
    """
    def __init__(self, base_dir, model_id):
        self._base_dir = base_dir
        self._dir = os.path.join(base_dir, model_id)
        self._summary_dir = os.path.join(self._dir, "tfsummary", self.__class__.__name__)
        self._sampling_dir = os.path.join(self._dir, "sampling", self.__class__.__name__)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError


class TrainableModel(BaseModel):
    """
    Iteratively trained model
    """
    def __init__(self, base_dir, model_id, training, ckpt=None):
        super(TrainableModel, self).__init__(base_dir, model_id)
        self.training = training
        self._ckpt = ckpt

        # ----- Directory Flags ----- #
        self._checkpoint_dir = os.path.join(self._dir, "checkpoints", self.__class__.__name__)

        # ----- Summary Writer ----- #
        if self.training:
            self._writer = tf.summary.create_file_writer(self._summary_dir)
        else:
            None

        # ----- Build Model ----- #
        self._build_model()

        # ----- Checkpoint Model ----- #
        self._checkpoint_model()

    def _build_model(self):
        raise NotImplementedError

    def _checkpoint_model(self):
        raise NotImplementedError

    def train(self, train_dataset, train_steps, print_freq, save_freq,  eval_dataset=None, eval_freq=None):
        raise NotImplementedError

    def evaluate(self, step, eval_dataset):
        raise NotImplementedError

    def test(self, test_dataset, result_name, steps=None):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _write_summaries(self, step, summaries_dict):
        for key in summaries_dict:
            tf.summary.scalar(key, summaries_dict[key], step=step)
