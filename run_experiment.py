import os
import traceback

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

import models
import configs
import datasets
from util import HParams
from util import log_flags, log_hparams

try:
    import horovod.tensorflow as hvd
except:
    hvd = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = flags.FLAGS

flags.DEFINE_string("dir", "/h/wangale/project/few-shot-sketch", "Project directory")
flags.DEFINE_string("data_dir", "/h/wangale/data", "Data directory")
flags.DEFINE_boolean("check_numerics", False, "Enable tensorflow check numerics.")

flags.DEFINE_string("id", None, "training_id")
flags.DEFINE_string("logfile", None, "Logfile name")

flags.DEFINE_string("model", None, "Model to train")
flags.DEFINE_string("model_cfgset", None, "Configuration set for the model")
flags.DEFINE_string("model_cfgs", "", "Custom configuration for the model configs")

flags.DEFINE_string("train_dataset", None, "Dataset used")
flags.DEFINE_string("train_dataset_cfgset", None, "Configuration set for the dataset")
flags.DEFINE_string("train_dataset_cfgs", "", "Custom configuration for the dataset configs")

flags.DEFINE_string("eval_dataset", None, "Dataset used")
flags.DEFINE_string("eval_dataset_cfgset", None, "Configuration set for the dataset")
flags.DEFINE_string("eval_dataset_cfgs", "", "Custom configuration for the dataset configs")

flags.DEFINE_integer("train_steps", None, "Training iterations")
flags.DEFINE_integer("print_freq", 250, "Training loop print frequency")
flags.DEFINE_integer("save_freq", 10000, "Save checkpoint frequency")
flags.DEFINE_integer("eval_freq", None, "Evaluation frequency; set to save_freq if None")
flags.DEFINE_boolean("distributed", False, "Distributed training if model architecture allows")

flags.DEFINE_integer("random_seed", 0, "Random seed")

flags.mark_flags_as_required(["id", "logfile",
                              "model", "model_cfgset",
                              "train_dataset", "train_dataset_cfgset",
                              "train_steps"])

def experiment():
    model_config: HParams = configs.get_config(FLAGS.model_cfgset)().parse(FLAGS.model_cfgs)
    model = models.get_model(FLAGS.model)(FLAGS.dir, FLAGS.id, model_config)

    train_dataset_config: HParams = configs.get_config(FLAGS.train_dataset_cfgset)().parse(FLAGS.train_dataset_cfgs)
    train_dataset = datasets.get_dataset(FLAGS.train_dataset)(FLAGS.data_dir, train_dataset_config)
    train_tf_dataset = train_dataset.load(repeat=True)

    if FLAGS.eval_dataset:
        eval_dataset_config: HParams = configs.get_config(FLAGS.eval_dataset_cfgset)().parse(FLAGS.eval_dataset_cfgs)
        eval_dataset = datasets.get_dataset(FLAGS.eval_dataset)(FLAGS.data_dir, eval_dataset_config)
        eval_tf_dataset = eval_dataset.load(repeat=False)
    else:
        eval_dataset_config = None
        eval_tf_dataset = None

    if (not FLAGS.distributed) or (hvd.rank() == 0):
        logging.info("Creating Model: %s | Loading Train Dataset: %s | Loading Eval Dataset: %s",
                     FLAGS.model, FLAGS.train_dataset, FLAGS.eval_dataset)
        log_hparams(model_config, train_dataset_config, eval_dataset_config)
        logging.info("Beginning training loop")

    # Debugging NaN errors.
    if FLAGS.check_numerics:
        tf.debugging.enable_check_numerics()

    while True:
        try:
            model.train(train_tf_dataset,
                        FLAGS.train_steps,
                        FLAGS.print_freq,
                        FLAGS.save_freq,
                        eval_tf_dataset,
                        FLAGS.eval_freq)
        except tf.errors.AbortedError:
            logging.info("InvalidArgumentError received from training function. Restarting training.")
            continue
        else:
            break


def main(argv):
    """Create directories and configure python settings"""

    # Setup Directory
    experiment_dir = os.path.join(FLAGS.dir, FLAGS.id)
    if not os.path.exists(experiment_dir):
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # Setup Logging
    FLAGS.alsologtostderr = True
    logging.get_absl_handler().use_absl_log_file(FLAGS.logfile, os.path.join(experiment_dir, "logs"))

    # Setup Distributed
    if FLAGS.distributed:
        try:
            hvd.init()
            gpus = tf.config.list_physical_devices('GPU')
            logging.info("Distributed training enabled.")
            logging.info("GPUS: %s", str(gpus))
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            FLAGS.model_cfgs = (FLAGS.model_cfgs + ",distributed=True").strip(',')
        except:
            logging.info("Distributed training training setup failed. Disabling distributed training.")
        if FLAGS.random_seed:
            logging.info("Setting seed to %s", FLAGS.random_seed + hvd.rank())
            np.random.seed(FLAGS.random_seed + hvd.rank())
            tf.random.set_seed(FLAGS.random_seed + hvd.rank())
    else:
        # Setup seeds
        if FLAGS.random_seed:
            logging.info("Setting seed to %s", FLAGS.random_seed)
            np.random.seed(FLAGS.random_seed)
            tf.random.set_seed(FLAGS.random_seed)

    # Log Flags
    if (not FLAGS.distributed) or (hvd.rank() == 0):
        log_flags(FLAGS)

    try:
        experiment()
    except:
        exception = traceback.format_exc()
        logging.info(exception)


if __name__ == "__main__":
    app.run(main)
