from configs.base import register_config

from util import HParams, teacher_noise_4, rotate_4


@register_config('quickdraw')
def quickdraw_default():
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

@register_config('quickdraw/batch128')
def quickdraw_batch128(hparams: HParams):
    hparams.set_hparam("batch_size", 128)

    return hparams

@register_config("quickdraw/noisy")
def quickdraw_noisy(hparam: HParams):
    try:
        hparam.add_hparam("augmentations", [[teacher_noise_4]])
    except:
        hparam.set_hparam("augmentations", hparam.augmentations.append(teacher_noise_4))

    return hparam

@register_config("quickdraw/rotate")
def quickdraw_rotate(hparam: HParams):
    try:
        hparam.add_hparam("augmentations", [[rotate_4]])
    except:
        hparam.set_hparam("augmentations", hparam.augmentations.append(rotate_4))
    return hparam

    return hparams

