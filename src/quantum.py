import numpy as np

from src.vectors import magnitude

def basis_zero() -> np.ndarray:
    """Returns the |0> computational basis state as a column vector."""
    return np.array([[1], [0]], dtype=complex)

def basis_one() -> np.ndarray:
    """Returns the |1> computational basis state as a column vector."""
    return np.array([[0], [1]], dtype=complex)

def normalise_state(state: np.ndarray) -> np.ndarray:
    """Returns a normalised quantum state vector (or an error if the state is the zero vector)."""
    norm = magnitude(state)
    if norm == 0:
        raise ValueError("Cannot normalise the zero vector.")
    return state / norm

def measurement_probabilities(state: np.ndarray) -> np.ndarray:
    """Returns the measurement probabilities of a quantum state via the Born rule."""
    state = normalise_state(state)
    return np.abs(state) ** 2