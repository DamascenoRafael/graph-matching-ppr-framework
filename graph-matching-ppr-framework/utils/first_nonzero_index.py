import numpy as np


def first_nonzero_index(sequence_a: list[np.float64], sequence_b: list[np.float64]):
    for index, (value_a, value_b) in enumerate(zip(sequence_a, sequence_b)):
        if value_a != 0 or value_b != 0:
            return index
    return None
