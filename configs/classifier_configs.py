from configs.base import register_config
from util import HParams, ST2_classes, T1_classes, T2_classes


@register_config("classifier")
def classifier_default():
    return HParams(
        # ----- Model Parameters ----- #
        class_list=None,
        png_dims=32,
        weights=None,

        # ----- Training Parameters ----- #
        lr=0.01,
        lr_schedule={40000: 0.005, 80000: 0.001, 100000: 0.0001}
    )

@register_config("classifier/T1")
def classifier_T1(hparam: HParams):

    hparam.set_hparam("class_list", ",".join(T1_classes))

    return hparam


@register_config("classifier/T2")
def classifier_T2(hparam: HParams):

    hparam.set_hparam("class_list", ",".join(T2_classes))

    return hparam
