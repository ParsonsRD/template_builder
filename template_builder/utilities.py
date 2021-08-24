import numpy as np

__all__ = ["find_nearest_bin"]

def find_nearest_bin(array, value):
    """
    Find nearest value in an array

    :param array: ndarray
        Array to search
    :param value: float
        Search value
    :return: float
        Nearest bin value
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx]