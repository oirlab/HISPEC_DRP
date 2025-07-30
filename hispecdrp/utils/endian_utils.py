import numpy as np

__all__ = ['normalize_dtype_array']

def normalize_dtype_array(arr):
    """
    Normalizes the data type of a NumPy array.

    Args:
        arr (np.ndarray): The input NumPy array.

    Returns:
        np.ndarray: The array with a normalized data type.
    """
    return arr.astype(np.dtype(arr.dtype).type, copy=False)