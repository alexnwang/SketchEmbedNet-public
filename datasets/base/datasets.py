DATA_SETS = {}


def register_dataset(dataset_name):
    """
    Decorator for registering a dataset class.
    :param dataset_name:
    :return:
    """
    def decorator(f):
        DATA_SETS[dataset_name] = f
        return f

    return decorator


def get_dataset(dataset):
    """
    Returns dataset class if registered.
    :param dataset:
    :return:
    """
    if dataset in DATA_SETS:
        dataset = DATA_SETS[dataset]

        return dataset
    else:
        raise ValueError("Dataset not found: %s", dataset)
