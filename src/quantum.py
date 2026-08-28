import numpy as np

from src.vectors import magnitude

"""Returns the |0> computational basis state as a column vector."""
def basis_zero() -> np.ndarray:
    return np.array([[1], [0]], dtype=complex)

"""Returns the |1> computational basis state as a column vector."""
def basis_one() -> np.ndarray:
    return np.array([[0], [1]], dtype=complex)

"""Returns a normalised quantum state vector (or an error if the state is the zero vector). """
def normalise_state(state: np.ndarray) -> np.ndarray:
    norm = magnitude(state)
    if norm == 0:
        raise ValueError("Cannot normalise the zero vector.")
    return state / norm

"""Returns the measurement probabilities of a quantum state via the Born rule."""
def measurement_probabilities(state: np.ndarray) -> np.ndarray:
    state = normalise_state(state)
    return np.abs(state) ** 2

"""Returns the Pauli-X matrix."""
def pauli_x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=complex)

"""Returns the Pauli-Y matrix."""
def pauli_y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

"""Returns the Pauli-Z matrix."""
def pauli_z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=complex)

"""Applies a quantum gate to a quantum state."""
def apply_gate(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    return gate @ state 

"""Returns the Hadamard matrix."""
def hadamard() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)