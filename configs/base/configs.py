from absl import logging

CONFIGS = {}


def register_config(config_name):
    """
    Decorator for registering config class.
    :param config_name:
    :return:
    """
    def decorator(f):
        CONFIGS[config_name] = f
        return f

    return decorator


def get_config_old(config):
    """
    Returns config class if registered.
    :param config:
    :return:
    """
    if config in CONFIGS:
        config = CONFIGS[config]

        return config
    else:
        raise ValueError("Config not found: %s", config)

def get_config(config: str):
    """
    Splits requested config string by '/', the base config is the first element in the list and every compounding modification
    is retrieved by passing the previous config into "base/mod".

    To keep in convention that we return a function that can be called for the config, we will use a lambda after generating the config.
    :param config:
    :return: fn: () -> config
    """

    terms = config.split('/')
    base, mods = terms[0], terms[1:]

    try:
        lookup = base
        config = CONFIGS[lookup]()

        for mod in mods:
            lookup = "{}/{}".format(base, mod)
            config = CONFIGS[lookup](config)

    except KeyError:
        logging.fatal("Incorrect config requested %s from %s", lookup, config)

    return lambda: config
