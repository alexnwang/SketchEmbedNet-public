from configs.base import register_config
from util import HParams, rotate_4, teacher_noise_4


@register_config('fs_omniglot')
def fs_omniglot_default():
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

@register_config("fs_omniglot/noisy")
def fs_omniglot_noisy(hparam: HParams):
    try:
        hparam.add_hparam("augmentations", [[teacher_noise_4]])
    except:
        hparam.set_hparam("augmentations", hparam.augmentations.append(teacher_noise_4))

    return hparam

@register_config("fs_omniglot/rotate")
def fs_omniglot_rotate(hparam: HParams):
    try:
        hparam.add_hparam("augmentations", [[rotate_4]])
    except:
        hparam.set_hparam("augmentations", hparam.augmentations.append(rotate_4))
    return hparam

@register_config("fs_omniglot/20way1shot")
def fs_omniglot_20way1shot(hparams: HParams):
    hparams.set_hparam("way", 20)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/20way5shot")
def fs_omniglot_20way5shot(hparams: HParams):
    hparams.set_hparam("way", 20)
    hparams.set_hparam("shot", 5)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/5way1shot")
def fs_omniglot_5way1shot(hparams: HParams):
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 1)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/5way5shot")
def fs_omniglot_5way5shot(hparams: HParams):
    hparams.set_hparam("way", 5)
    hparams.set_hparam("shot", 5)
    hparams.set_hparam("mode", "episodic")

    return hparams

@register_config("fs_omniglot/vinyals_test_fake")
def fs_omniglot_vinyals_test_fake(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Gurmukhi/character42.npz,Gurmukhi/character43.npz,Gurmukhi/character44.npz,Gurmukhi/character45.npz,"
                       + "Kannada,Keble,Malayalam,Manipuri,Mongolian,Old_Church_Slavonic_(Cyrillic),Oriya,Syriac_(Serto),Sylheti,"
                       + "Tengwar,Tibetan,ULOG")

    return hparams

@register_config("fs_omniglot/vinyals_train_fake")
def fs_omniglot_vinyals_train_fake(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Angelic,Grantha,N_Ko,Aurek-Besh,Japanese_(hiragana),Malay_(Jawi_-_Arabic),Asomtavruli_(Georgian),Sanskrit,"
                       + "Ojibwe_(Canadian_Aboriginal_Syllabics),Korean,Arcadian,Greek,Alphabet_of_the_Magi,"
                       + "Blackfoot_(Canadian_Aboriginal_Syllabics),Futurama,Gurmukhi/character01.npz,Gurmukhi/character02.npz,"
                       + "Gurmukhi/character03.npz,Gurmukhi/character04.npz,Gurmukhi/character05.npz,Gurmukhi/character06.npz,"
                       + "Gurmukhi/character07.npz,Gurmukhi/character08.npz,Gurmukhi/character09.npz,Gurmukhi/character10.npz,"
                       + "Gurmukhi/character11.npz,Gurmukhi/character12.npz,Gurmukhi/character13.npz,Gurmukhi/character14.npz,"
                       + "Gurmukhi/character15.npz,Gurmukhi/character16.npz,Gurmukhi/character17.npz,Gurmukhi/character18.npz,"
                       + "Gurmukhi/character19.npz,Gurmukhi/character20.npz,Gurmukhi/character21.npz,Gurmukhi/character22.npz,"
                       + "Gurmukhi/character23.npz,Gurmukhi/character24.npz,Gurmukhi/character25.npz,Gurmukhi/character26.npz,"
                       + "Gurmukhi/character27.npz,Gurmukhi/character28.npz,Gurmukhi/character29.npz,Gurmukhi/character30.npz,"
                       + "Gurmukhi/character31.npz,Gurmukhi/character32.npz,Gurmukhi/character33.npz,Gurmukhi/character34.npz,"
                       + "Gurmukhi/character35.npz,Gurmukhi/character36.npz,Gurmukhi/character37.npz,Gurmukhi/character38.npz,"
                       + "Gurmukhi/character39.npz,Gurmukhi/character40.npz,Gurmukhi/character41.npz,Tagalog,Anglo-Saxon_Futhorc,"
                       + "Braille,Cyrillic,Burmese_(Myanmar),Avesta,Gujarati,Ge_ez,Syriac_(Estrangelo),Atlantean,"
                       + "Japanese_(katakana),Balinese,Atemayar_Qelisayer,Glagolitic,Tifinagh,Latin,"
                       + "Inuktitut_(Canadian_Aboriginal_Syllabics)")
    hparams.set_hparam("shuffle", True)
    return hparams

@register_config("fs_omniglot/lake_test")
def fs_omniglot_lake_test(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Angelic,Atemayar_Qelisayer,Atlantean,Aurek-Besh,Avesta,Ge_ez,Glagolitic,Gurmukhi,Kannada,Keble,"
                                + "Malayalam,Manipuri,Mongolian,Old_Church_Slavonic_(Cyrillic),Oriya,Sylheti,Syriac_(Serto),Tengwar,"
                                + "Tibetan,ULOG")

    return hparams

@register_config("fs_omniglot/lake_train")
def fs_omniglot_lake_train(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Alphabet_of_the_Magi,Anglo-Saxon_Futhorc,Arcadian,Armenian,Asomtavruli_(Georgian),Balinese,Bengali,"
                                "Blackfoot_(Canadian_Aboriginal_Syllabics),Braille,Burmese_(Myanmar),Cyrillic,Early_Aramaic,Futurama,"
                                "Grantha,Greek,Gujarati,Hebrew,Inuktitut_(Canadian_Aboriginal_Syllabics),Japanese_(hiragana),"
                                "Japanese_(katakana),Korean,Latin,Malay_(Jawi_-_Arabic),Mkhedruli_(Georgian),N_Ko,"
                                "Ojibwe_(Canadian_Aboriginal_Syllabics),Sanskrit,Syriac_(Estrangelo),Tagalog,Tifinagh")

    hparams.set_hparam("shuffle", True)
    hparams.set_hparam("cycle_length", 50)

    return hparams

@register_config("fs_omniglot/vinyals_val")
def fs_omniglot_vinyals_val(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Hebrew,Mkhedruli_(Georgian),Armenian,Early_Aramaic,Bengali")

    return hparams

@register_config("fs_omniglot/vinyals_train")
def fs_omniglot_vinyals_train(hparams: HParams):
    hparams.set_hparam("mode", "batch")

    file_list = "Gurmukhi/character01.npz,Gurmukhi/character02.npz,Gurmukhi/character03.npz,Gurmukhi/character04.npz,Gurmukhi/character05.npz,Gurmukhi/character06.npz,Gurmukhi/character07.npz,Gurmukhi/character08.npz,Gurmukhi/character09.npz,Gurmukhi/character10.npz,Gurmukhi/character11.npz,Gurmukhi/character12.npz,Gurmukhi/character13.npz,Gurmukhi/character14.npz,Gurmukhi/character15.npz,Gurmukhi/character16.npz,Gurmukhi/character17.npz,Gurmukhi/character18.npz,Gurmukhi/character19.npz,Gurmukhi/character20.npz,Gurmukhi/character21.npz,Gurmukhi/character22.npz,Gurmukhi/character23.npz,Gurmukhi/character24.npz,Gurmukhi/character25.npz,Gurmukhi/character26.npz,Gurmukhi/character27.npz,Gurmukhi/character28.npz,Gurmukhi/character29.npz,Gurmukhi/character30.npz,Gurmukhi/character31.npz,Gurmukhi/character32.npz,Gurmukhi/character33.npz,Gurmukhi/character34.npz,Gurmukhi/character35.npz,Gurmukhi/character36.npz,Gurmukhi/character37.npz,Gurmukhi/character38.npz,Gurmukhi/character39.npz,Gurmukhi/character40.npz,Gurmukhi/character41.npz".split(
        ",")
    new_list = []
    for file in file_list:
        pre_ext, post_ext = file.split(".")
        for rot in ["", "-rot90", "-rot180", "-rot270"]:
            new_list.append("{}{}.{}".format(pre_ext, rot, post_ext))
    file_str = ",".join(new_list)
    file_str = file_str + (",Angelic,Grantha,N_Ko,Aurek-Besh,Japanese_(hiragana),Malay_(Jawi_-_Arabic),Asomtavruli_(Georgian),Sanskrit,"
                           + "Ojibwe_(Canadian_Aboriginal_Syllabics),Korean,Arcadian,Greek,Alphabet_of_the_Magi,"
                           + "Blackfoot_(Canadian_Aboriginal_Syllabics),Futurama,Tagalog,Anglo-Saxon_Futhorc,"
                           + "Braille,Cyrillic,Burmese_(Myanmar),Avesta,Gujarati,Ge_ez,Syriac_(Estrangelo),Atlantean,"
                           + "Japanese_(katakana),Balinese,Atemayar_Qelisayer,Glagolitic,Tifinagh,Latin,"
                           + "Inuktitut_(Canadian_Aboriginal_Syllabics)")

    hparams.set_hparam("split", file_str)

    return hparams

@register_config("fs_omniglot/vinyals_test")
def fs_omniglot_vinyals_test(hparams: HParams):
    hparams.set_hparam("mode", "batch")
    hparams.set_hparam("split", "Gurmukhi/character42.npz,Gurmukhi/character43.npz,Gurmukhi/character44.npz,Gurmukhi/character45.npz,"
                                + "Kannada,Keble,Malayalam,Manipuri,Mongolian,Old_Church_Slavonic_(Cyrillic),Oriya,Syriac_(Serto),Sylheti,"
                                + "Tengwar,Tibetan,ULOG,"
                                + "Gurmukhi/character42-rot90.npz,Gurmukhi/character42-rot180.npz,Gurmukhi/character42-rot270.npz,"
                                + "Gurmukhi/character43-rot90.npz,Gurmukhi/character43-rot180.npz,Gurmukhi/character43-rot270.npz,"
                                + "Gurmukhi/character44-rot90.npz,Gurmukhi/character44-rot180.npz,Gurmukhi/character44-rot270.npz,"
                                + "Gurmukhi/character45-rot90.npz,Gurmukhi/character45-rot180.npz,Gurmukhi/character45-rot270.npz")

    return hparams
