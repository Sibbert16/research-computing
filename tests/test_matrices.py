import numpy as np

from src.matrices import (
    eigenvalues,
    eigenvectors,
    is_hermitian,
    is_unitary
)


def test_eigenvalues():
    A = np.array([[2, 0],
                  [0, 3]])

    result = eigenvalues(A)

    assert np.allclose(result, [2, 3])


def test_eigenvectors():
    A = np.array([[2, 0],
                  [0, 3]])

    result = eigenvectors(A)

    # Check that the output has the right shape
    assert result.shape == (2, 2)


def test_hermitian():
    A = np.array([[1, 2],
                  [2, 3]])

    assert is_hermitian(A)


def test_not_hermitian():
    A = np.array([[1, 2],
                  [3, 4]])

    assert not is_hermitian(A)


def test_unitary():
    X = np.array([[0, 1],
                  [1, 0]])

    assert is_unitary(X)