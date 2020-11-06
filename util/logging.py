import logging


def log_flags(flags):
    logging.info("---------------------------------------------FLAGS---------------------------------------------")
    flags_dict = flags.flag_values_dict()

    for key in list(flags_dict.keys())[20:-5]:
        logging.info("[%s]: %s", key, flags_dict[key])
    logging.info("-----------------------------------------------------------------------------------------------")


def log_hparams(*args):
    logging.info("-----------------------------------------------------------------------------------------------")
    for hparams in args:
        if not hparams:
            continue
        values = hparams.values()
        for key in values:
            logging.info("[%s]: %s", key, values[key])
        logging.info("-----------------------------------------------------------------------------------------------")


def bar():
    logging.info("-----------------------------------------------------------------------------------------------")