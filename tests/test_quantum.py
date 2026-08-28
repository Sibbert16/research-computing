import numpy as np

from src.quantum import (
    basis_zero,
    basis_one,
    normalise_state,
    measurement_probabilities
)

def test_basis_zero():
    expected = np.array([[1], [0]], dtype=complex)
    assert np.allclose(basis_zero(), expected)

def test_basis_one():
    expected = np.array([[0], [1]], dtype=complex)
    assert np.allclose(basis_one(), expected)

def test_normalise_state():
    state = np.array([[3], [4]], dtype=complex)
    expected = np.array([[0.6], [0.8]], dtype=complex)
    assert np.allclose(normalise_state(state), expected)

def test_measurement_probabilities():
    state = np.array([[3], [4]], dtype=complex)
    expected = np.array([[0.36], [0.64]], dtype=complex)
    assert np.allclose(measurement_probabilities(state), expected)