from configs.base import register_config

from util import HParams, teacher_noise_4, rotate_4

@register_config('sketchy')
def sketchy_default():
    return HParams(
        # ----- Dataset Parameters ----- #
        batch_size=256,
        split="",

        # ----- Loading Parameters ----- #
        cycle_length=None,
        num_parallel_calls=None,
        block_length=1,
        buff_size=2,
        shuffle=True,
    )

@register_config("sketchy/noisy")
def sketchy_noisy(hparam: HParams):
    try:
        hparam.add_hparam("augmentations", [[teacher_noise_4]])
    except:
        hparam.set_hparam("augmentations", hparam.augmentations.append(teacher_noise_4))

    return hparam

@register_config("sketchy/rotate")
def sketchy_rotate(hparam: HParams):
    try:
        hparam.add_hparam("augmentations", [[rotate_4]])
    except:
        hparam.set_hparam("augmentations", hparam.augmentations.append(rotate_4))
    return hparam

@register_config('sketchy/batch128')
def sketchy_batch128(hparams: HParams):
    hparams = sketchy_default()
    hparams.set_hparam("batch_size", 128)

    return hparams

@register_config('sketchy/batch64')
def sketchy_batch64(hparams: HParams):
    hparams.set_hparam("batch_size", 64)

    return hparams

@register_config("sketchy_batch64/rot")
def sketchy_batch64_rot():
    hparams = sketchy_batch64()
    hparams.add_hparam("augmentations", [[rotate_4]])
    return hparams

@register_config("sketchy_batch64/noise_rot")
def sketchy_batch64_noiserot():
    hparams = sketchy_batch64()
    hparams.add_hparam("augmentations", [[rotate_4, teacher_noise_4]])
    return hparams

@register_config("sketchy_batch64/noise")
def sketchy_batch64_noise():
    hparams = sketchy_batch64()
    hparams.add_hparam("augmentations", [[teacher_noise_4]])
    return hparams
