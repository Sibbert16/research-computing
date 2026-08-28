import numpy as np

from src.quantum import (
    basis_zero,
    basis_one,
    normalise_state,
    measurement_probabilities,
    pauli_x,
    pauli_y,
    pauli_z,
    hadamard,
    apply_gate
)

from src.matrices import (
    is_hermitian,
    is_unitary
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

def test_pauli_hermitian():
    x = is_hermitian(pauli_x())
    y = is_hermitian(pauli_y())
    z = is_hermitian(pauli_z())
    assert x == True
    assert y == True
    assert z == True

def test_pauli_unitary():
    x = is_unitary(pauli_x())
    y = is_unitary(pauli_y())
    z = is_unitary(pauli_z())
    assert x == True
    assert y == True
    assert z == True

def test_hadamard_unitary():
    h = is_unitary(hadamard())
    assert h == True

def test_apply_gate():
    state = np.array([[1], [0]], dtype=complex)
    expected = np.array([[0], [1]], dtype=complex)
    result = apply_gate(pauli_x(), state)
    assert np.allclose(result, expected)