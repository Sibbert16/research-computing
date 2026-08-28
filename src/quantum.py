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

def pauli_x() -> np.ndarray:
    """Returns the Pauli-X matrix."""
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_y() -> np.ndarray:
    """Returns the Pauli-Y matrix."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def pauli_z() -> np.ndarray:
    """Returns the Pauli-Z matrix."""
    return np.array([[1, 0], [0, -1]], dtype=complex)

def apply_gate(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Applies a quantum gate to a quantum state."""
    return gate @ state 