import numpy as np
from .normalize import normalize


def prepare_for_training(data, normalize_data=True):
    num_examples = data.shape[0]

    data_processed = np.copy(data)
    features_mean = 0
    features_deviation = 0
    if normalize_data:
        (data_normalized, features_mean, features_deviation) = normalize(data_processed)
        data_processed = data_normalized

    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))
    return data_processed, features_mean, features_deviation
