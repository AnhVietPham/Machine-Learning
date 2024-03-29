import numpy as np


def normalize(features):
    features_normalize = np.copy(features).astype(float)
    features_mean = np.mean(features, 0)
    features_deviation = np.std(features, 0)

    if features.shape[0] > 1:
        features_normalize -= features_mean

    features_deviation[features_deviation == 0] = 1
    features_normalize /= features_deviation
    return features_normalize, features_mean, features_deviation
