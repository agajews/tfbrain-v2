import numpy as np


def flatten_params(params):
    def sorted_dict(dictionary):
        items = sorted(dictionary.items(), key=lambda i: i[0])
        return [i[1] for i in items]
    return [p for l_p in sorted_dict(params) for p in sorted_dict(l_p)]


def iterate_minibatches(*arrays_dicts, batch_size=128, shuffle=True):
    first = arrays_dicts[0][list(arrays_dicts[0].keys())[0]]
    length = first.shape[0]
    for arrays_dict in arrays_dicts:
        for array_name, array in arrays_dict.items():
            assert array.shape[0] == length
    if shuffle:
        indices = np.arange(length)
        np.random.shuffle(indices)
    for start_index in range(0, length - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_index:start_index + batch_size]
        else:
            excerpt = slice(start_index, start_index + batch_size)
        batch = [{n: a[excerpt] for (n, a) in a_d.items()}
                 for a_d in arrays_dicts]
        if len(batch) == 1:
            batch = batch[0]
        yield batch


def create_minibatch_iterator(train_xs,
                              train_y,
                              batch_preprocessor=None,
                              batch_size=128,
                              shuffle=True):
    minibatches = iterate_minibatches(
        train_xs, train_y, batch_size=batch_size, shuffle=shuffle)

    if batch_preprocessor is None:
        return minibatches
    else:
        return map(batch_preprocessor, minibatches)


def avg_over_batches(xs, y, preprocessor, fn, batch_size=128):
    minibatches = create_minibatch_iterator(
        xs, {'y': y}, preprocessor,
        batch_size=batch_size)
    vals = [fn(x, y['y']) for (x, y) in minibatches]
    return sum(vals) / len(vals)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
