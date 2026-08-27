import numpy as np

from src.vectors import (
    magnitude,
    normalise,
    dot_product,
    add_vectors,
    scalar_multiply
)


def test_magnitude():
    v = np.array([3, 4])

    result = magnitude(v)

    assert result == 5


def test_normalise():
    v = np.array([3, 4])

    result = normalise(v)

    assert np.allclose(result, [0.6, 0.8])


def test_dot_product():
    a = np.array([1, 2])
    b = np.array([3, 4])

    result = dot_product(a, b)

    assert result == 11


def test_add_vectors():
    a = np.array([1, 2])
    b = np.array([3, 4])

    result = add_vectors(a, b)

    assert np.allclose(result, [4, 6])


def test_scalar_multiply():
    v = np.array([1, 2])

    result = scalar_multiply(3, v)

    assert np.allclose(result, [3, 6])