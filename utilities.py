import numpy as np


def create_patterns(name: str):
    dataset = list()
    with open(name) as datafile:
        for row in datafile:
            dataset.append(row)

    return np.asarray(dataset)
