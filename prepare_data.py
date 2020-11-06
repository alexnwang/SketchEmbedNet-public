import os
import traceback

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

import configs
import datasets
from util import HParams
from util import log_flags, log_hparams

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", "/h/wangale/project/few-shot-sketch", "Project directory")
flags.DEFINE_string("data_dir", "/h/wangale/data", "Data directory")

flags.DEFINE_string("id", None, "training_id")
flags.DEFINE_string("logfile", "", "Logfile name")
flags.DEFINE_boolean("test", False, "Perform testing")

flags.DEFINE_string("dataset", None, "Dataset used")
flags.DEFINE_string("dataset_cfgset", None, "Configuration set for the dataset")
flags.DEFINE_string("dataset_cfgs", "", "Custom configuration for the dataset configs")

flags.DEFINE_integer("random_seed", 1, "Random seed")

flags.mark_flags_as_required(["id", "dataset", "dataset_cfgset"])


def prepare():
    dataset_config: HParams = configs.get_config(FLAGS.dataset_cfgset)().parse(FLAGS.dataset_cfgs)
    log_hparams(dataset_config)

    logging.info("Getting and preparing dataset: %s", FLAGS.dataset)
    dataset = datasets.get_dataset(FLAGS.dataset)(FLAGS.data_dir, dataset_config)
    dataset.prepare(FLAGS)


def main(argv):
    """Create directories and configure python settings"""
    FLAGS.dir = os.path.join(FLAGS.dir, FLAGS.id)
    if not os.path.exists(FLAGS.dir):
        os.makedirs(FLAGS.dir)
        os.makedirs(os.path.join(FLAGS.dir, "logs"))

    FLAGS.alsologtostderr = True
    logging.get_absl_handler().use_absl_log_file(FLAGS.logfile, os.path.join(FLAGS.dir, "logs"))

    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)

    log_flags(FLAGS)

    try:
        prepare()
    except:
        exception = traceback.format_exc()
        logging.info(exception)


if __name__ == "__main__":
    app.run(main)
