from configs import register_config
from util import HParams, miniimagenet_train, miniimagenet_eval


@register_config("miniimagenet")
def miniimagenet_default():
    return HParams(
        # ----- Dataset Parameters ----- #
        split="",
        mode="batch",  # episodic or batch

        # ----- Batch Parameters ----- #
        batch_size=256,

        # ----- Episodic Parameters ----- #
        episodic=False,
        way=0,
        shot=0,

        # ----- Loading Parameters ----- #
        cycle_length=None,
        num_parallel_calls=None,
        block_length=1,
        buff_size=2,
        shuffle=False,
    )

@register_config("miniimagenet/5way1shot")
def miniimagenet_5way1shot(hparams: HParams):
    hparams.set_hparam("mode", "episodic")
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 1)

    return hparams

@register_config("miniimagenet/5way5shot")
def miniimagenet_5way5shot(hparams: HParams):
    hparams.set_hparam("mode", "episodic")
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 5)

    return hparams

@register_config("miniimagenet/5way20shot")
def miniimagenet_5way20shot(hparams: HParams):
    hparams.set_hparam("mode", "episodic")
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 20)

    return hparams

@register_config("miniimagenet/5way50shot")
def miniimagenet_5way5shot(hparams: HParams):
    hparams.set_hparam("mode", "episodic")
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 50)

    return hparams

@register_config("miniimagenet/sachinravi_train")
def miniimagenet_sachinravi_train(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split",
                       ".npz,".join(miniimagenet_train)+".npz")
    hparams.set_hparam("shuffle", True)
    return hparams

@register_config("miniimagenet/sachinravi_val")
def miniimagenet_sachinravi_val(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split",
                       ".npz,".join(miniimagenet_eval)+".npz")
    hparams.set_hparam("shuffle", True)
    return hparams

@register_config("miniimagenet/sachinravi_test")
def miniimagenet_sachinravi_test(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", 'n01981276.npz,n02116738.npz,n03146219.npz,n04149813.npz,n04146614.npz,n04522168.npz,n02099601.npz,n02443484.npz,n02129165.npz,n03272010.npz,'
                                'n04418357.npz,n03127925.npz,n02110063.npz,n02871525.npz,n03775546.npz,n02219486.npz,n02110341.npz,n07613480.npz,n03544143.npz,n01930112.npz')
    hparams.set_hparam("shuffle", True)
    return hparams