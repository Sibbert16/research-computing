import numpy as np


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Multiply two matrices together.

    Parameters:
        A: First matrix.
        B: Second matrix.

    Returns:
        The matrix product AB.
    """
    return A @ B