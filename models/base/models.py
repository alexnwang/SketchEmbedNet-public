MODELS = {}


def register_model(model_name):
    """
    Decorator for registering a model class.
    :param model_name:
    :return:
    """
    def decorator(f):
        MODELS[model_name] = f
        return f

    return decorator


def get_model(model):
    """
    Returns model class if registered.
    :param model:
    :return:
    """
    if model in MODELS:
        model = MODELS[model]

        return model
    else:
        raise ValueError("Model not found: {}".format(model))
