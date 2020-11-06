import os
import traceback

import numpy as np
import tensorflow as tf
from absl import app, flags, logging

import models
import configs
import datasets
from models import ClassifierModel, DrawerModel
from util import HParams, interpolate
from util import log_flags

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
flags.DEFINE_integer("ckpt", None, "checkpoint")

flags.DEFINE_string("class_model_id", None, "Classifier model ID")
flags.DEFINE_string("class_model", None, "Model to train")
flags.DEFINE_string("class_model_cfgset", None, "Configuration set for the model")
flags.DEFINE_string("class_model_cfgs", "", "Custom configuration for the model configs")

flags.DEFINE_boolean("natural", False, "Natural image model")
flags.DEFINE_boolean("sample", True, "Sample generated images")
flags.DEFINE_boolean("usfs", True, "Test unsupervised few-shot classification")
flags.DEFINE_boolean("checkpoint", True, "Test model over checkpoints")
flags.DEFINE_boolean("gen", True, "Test model generation")

flags.DEFINE_integer("random_seed", 0, "Random seed")

flags.mark_flags_as_required(["id", "logfile",
                              "model", "model_cfgset"])

def experiment():
    model_config: HParams = configs.get_config(FLAGS.model_cfgset)().parse(FLAGS.model_cfgs)
    model = models.get_model(FLAGS.model)(FLAGS.dir, FLAGS.id, model_config, training=False, ckpt=FLAGS.ckpt)

    if FLAGS.natural:
        # If natural images, use miniImageNet.
        logging.info("#######################################################")
        logging.info("#########Natural Images, using mini-ImageNet###########")
        logging.info("#######################################################")
        if FLAGS.sample:
            logging.info("=======================================================")
            logging.info("============Sampling examples from datasets============")
            logging.info("=======================================================")
            logging.info("=====Sampling Decodings of mini-ImageNet Examples...=====")
            sample_dataset_config: HParams = configs.get_config("miniimagenet/sachinravi_test")().parse("")
            gen_dataset_proto = datasets.get_dataset("miniimagenet")(FLAGS.data_dir, sample_dataset_config)
            sample_dataset = gen_dataset_proto.load(repeat=False)

            model.test(sample_dataset,
                       "full-eval-mii",
                       40,
                       generation_length=100)

            logging.info("=====Sampling Decodings of Sketchy Examples...=====")

            sample_dataset_config: HParams = configs.get_config("sketchy")().parse("split=msl100_84_noclash_noflip,shuffle=False")
            gen_dataset_proto = datasets.get_dataset("sketchy")(FLAGS.data_dir, sample_dataset_config)
            sample_dataset = gen_dataset_proto.load(repeat=False)

            model.test(sample_dataset,
                       "full-eval-sketchy",
                       20,
                       generation_length=100)

        if FLAGS.usfs:
            logging.info("=======================================================")
            logging.info("==========Unsupervised few-shot mini-ImageNet==========")
            logging.info("=======================================================")

            for cmodel_type in ["lr_fs"]:
                cmodel_config = configs.get_config("lr_fs")().parse("")
                cmodel = models.get_model(cmodel_type)(FLAGS.dir, FLAGS.id, cmodel_config)
                for split in ["sachinravi"]:
                    for setup in ["5way1shot", "5way5shot", "5way20shot", "5way50shot"]:
                        usfs_dataset_config = configs.get_config("miniimagenet/{}_test/{}".format(split, setup))().parse("")
                        usfs_dataset = datasets.get_dataset("miniimagenet")(FLAGS.data_dir, usfs_dataset_config)

                        logging.info("===== Running Unsupervised Few-shot | linear_head: %s | split: %s | %s  ======", cmodel_type, split,
                                     setup)
                        cmodel.episode(model, usfs_dataset, 1000)

        if FLAGS.checkpoint:
            logging.info("================================================")
            logging.info("==========Performing checkpoint sweep.==========")
            logging.info("================================================")
            ckpts_dir = os.path.join(FLAGS.dir, FLAGS.id, "checkpoints")
            ckpts = os.listdir(os.path.join(ckpts_dir, os.listdir(ckpts_dir)[0]))

            ckpts = list(filter(lambda x: x.endswith(".index"), ckpts))
            ckpt_ids = [str(y) for y in sorted(list(map(lambda x: int(x.split(".")[0].split("-")[-1]), ckpts)))]

            for ckpt_id in ckpt_ids:
                logging.info("=====Loading Model with ckpt %s=====", ckpt_id)
                model_config: HParams = configs.get_config(FLAGS.model_cfgset)().parse(FLAGS.model_cfgs)
                model = models.get_model(FLAGS.model)(FLAGS.dir, FLAGS.id, model_config, training=False, ckpt=ckpt_id)

                for cmodel_type in ["lr_fs"]:
                    cmodel_config = HParams().parse("")
                    cmodel = models.get_model(cmodel_type)(FLAGS.dir, FLAGS.id, cmodel_config)
                    for split in ["sachinravi"]:
                        for setup in ["5way1shot"]:
                            usfs_dataset_config = configs.get_config("miniimagenet/{}_test/{}".format(split, setup))().parse("")
                            usfs_dataset = datasets.get_dataset("miniimagenet")(FLAGS.data_dir, usfs_dataset_config)

                            logging.info("===== Running Unsupervised Few-shot | linear_head: %s | split: %s | %s  ======", cmodel_type,
                                         split, setup)
                            cmodel.episode(model, usfs_dataset, 500)
    else:
        logging.info("#######################################################")
        logging.info("########## Sketches using Omniglot dataset ############")
        logging.info("#######################################################")
        ds = "fs_omniglot_28"
        if FLAGS.sample:
            logging.info("============================")
            logging.info("=====Sampling decodings=====")
            logging.info("============================")

            logging.info("=====Omniglot dataset=====")
            sample_dataset_config: HParams = configs.get_config("fs_omniglot/vinyals_test_fake")().parse("")
            gen_dataset_proto = datasets.get_dataset(ds)(FLAGS.data_dir, sample_dataset_config)
            sample_dataset = gen_dataset_proto.load(repeat=False)
            model.test(sample_dataset,
                       "full-eval-sample",
                       40)

            logging.info("=====Seen quickdraw examples=====")
            sample1_dataset_config: HParams = configs.get_config("quickdraw")().parse("split=T1_msl64_28,shuffle=False")
            gen1_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, sample1_dataset_config)
            sample1_dataset = gen1_dataset_proto.load(repeat=False)
            model.test(sample1_dataset,
                       "full-eval-qd-T1",
                       40)

            logging.info("=====Unseen quickdraw examples=====")
            sample2_dataset_config: HParams = configs.get_config("quickdraw")().parse("split=T2_msl64_28,shuffle=False")
            gen2_dataset_proto = datasets.get_dataset('quickdraw')(FLAGS.data_dir, sample2_dataset_config)
            sample2_dataset = gen2_dataset_proto.load(repeat=False)
            model.test(sample2_dataset,
                       "full-eval-qd-T2",
                       40)

            logging.info("=====Latent Space Interpolation=====")
            if isinstance(model, DrawerModel):
                interpolate(model,
                            sample1_dataset,
                            "interpolations",
                            interps=20)

        if FLAGS.gen:
            logging.info("=======================================")
            logging.info("==========Generation Testing===========")
            logging.info("=======================================")

            logging.info("===== Classifier Dataset: T1 =====")
            classifier1_configs = configs.get_config("classifier/T1")().parse("")
            class_model1: ClassifierModel = models.get_model("classifier")(FLAGS.dir, "05-24_classifiers/classifier_T1",
                                                                           classifier1_configs, training=False)

            gen1_dataset_config: HParams = configs.get_config("quickdraw")().parse("split=T1_msl64_28")
            gen1_dataset_proto = datasets.get_dataset("quickdraw")(FLAGS.data_dir, gen1_dataset_config)
            gen1_dataset = gen1_dataset_proto.load(repeat=False)

            logging.info("ST1 Classifier Test")
            class_model1.classify_predictions(gen1_dataset, model, steps=20)

            logging.info("===== Classifier Dataset: T2 =====")
            classifier2_configs = configs.get_config("classifier/T2")().parse("")
            class_model2: ClassifierModel = models.get_model("classifier")(FLAGS.dir, "05-24_classifiers/classifier_T2",
                                                                           classifier2_configs, training=False)

            gen2_dataset_config: HParams = configs.get_config("quickdraw")().parse("split=T2_msl64_28")
            gen2_dataset_proto = datasets.get_dataset("quickdraw")(FLAGS.data_dir, gen2_dataset_config)
            gen2_dataset = gen2_dataset_proto.load(repeat=False)

            logging.info("ST2 Classifier Test")
            class_model2.classify_predictions(gen2_dataset, model, steps=20)


        if FLAGS.usfs:
            logging.info("==================================================")
            logging.info("==========Unsupervised few-shot Omniglot==========")
            logging.info("==================================================")

            for cmodel_type in ["lr_fs"]:
                cmodel_config = HParams().parse("")
                cmodel = models.get_model(cmodel_type)(FLAGS.dir, FLAGS.id, cmodel_config)
                for setup in ["20way1shot", "20way5shot", "5way1shot", "5way5shot"]:
                    usfs_dataset_config = configs.get_config("fs_omniglot/vinyals_test/{}".format(setup))().parse("")
                    usfs_dataset = datasets.get_dataset("fs_omniglot_vinyals")(FLAGS.data_dir, usfs_dataset_config)

                    logging.info("===== Running Unsupervised Few-shot | linear_head: %s | split: %s | %s  ======", cmodel_type,
                                 "vinyals", setup)
                    cmodel.episode(model, usfs_dataset, 2000)

                    for split in ["lake"]:
                        logging.info("===Getting usfs test dataset: %s/%s===", split, setup)
                        usfs_dataset_config = configs.get_config("fs_omniglot/{}_test/{}".format(split, setup))().parse("")
                        usfs_dataset = datasets.get_dataset(ds)(FLAGS.data_dir, usfs_dataset_config)

                        logging.info("===== Running Unsupervised Few-shot | linear_head: %s | split: %s | %s  ======", cmodel_type,
                                     split, setup)
                        cmodel.episode(model, usfs_dataset, 2000)

        if FLAGS.checkpoint:
            logging.info("================================================")
            logging.info("==========Performing checkpoint sweep.==========")
            logging.info("================================================")
            ckpts_dir = os.path.join(FLAGS.dir, FLAGS.id, "checkpoints")
            ckpts = os.listdir(os.path.join(ckpts_dir, os.listdir(ckpts_dir)[0]))

            ckpts = list(filter(lambda x: x.endswith(".index"), ckpts))
            ckpt_ids = sorted(list(map(lambda x: int(x.split(".")[0].split("-")[-1]), ckpts)))

            for ckpt_id in ckpt_ids:
                ckpt_id = str(ckpt_id)
                logging.info("=====Loading Model with ckpt %s=====", ckpt_id)
                ckpt_model_config: HParams = configs.get_config(FLAGS.model_cfgset)().parse(FLAGS.model_cfgs)
                ckpt_model = models.get_model(FLAGS.model)(FLAGS.dir, FLAGS.id, ckpt_model_config, training=False, ckpt=ckpt_id)

                for cmodel_type in ["lr_fs"]:
                    cmodel_config = HParams().parse("")
                    cmodel = models.get_model(cmodel_type)(FLAGS.dir, FLAGS.id, cmodel_config)
                    for setup in ["20way1shot"]:
                        usfs_dataset_config = configs.get_config("fs_omniglot/vinyals_test/{}".format(setup))().parse("")
                        usfs_dataset = datasets.get_dataset("fs_omniglot_vinyals")(FLAGS.data_dir, usfs_dataset_config)

                        logging.info("===== Running Unsupervised Few-shot | linear_head: %s | split: %s | %s  ======", cmodel_type,
                                     "Vinyals", setup)
                        cmodel.episode(ckpt_model, usfs_dataset, 500)

                if FLAGS.gen:
                    logging.info("==========Generation Testing===========")
                    logging.info("ST1")
                    class_model1.classify_predictions(gen1_dataset, ckpt_model, steps=5)
                    logging.info("ST2")
                    class_model2.classify_predictions(gen2_dataset, ckpt_model, steps=5)


def main(argv):
    """Create directories and configure python settings"""

    # Setup Directory
    experiment_dir = os.path.join(FLAGS.dir, FLAGS.id)
    if not os.path.exists(experiment_dir):
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # Setup Logging
    FLAGS.alsologtostderr = True
    logging.get_absl_handler().use_absl_log_file(FLAGS.logfile, os.path.join(experiment_dir, "logs"))

    # Setup seeds
    if FLAGS.random_seed:
        np.random.seed(FLAGS.random_seed)
        tf.random.set_seed(FLAGS.random_seed)

    # Log Flags
    log_flags(FLAGS)

    try:
        experiment()
    except:
        exception = traceback.format_exc()
        logging.info(exception)


if __name__ == "__main__":
    app.run(main)
