import numpy as np

"""Returns magnitude of vector"""
def magnitude(vector: np.ndarray) -> float:
    return np.linalg.norm(vector)

"""Returns normalisation of vector"""
def normalise(vector: np.ndarray) -> np.ndarray:
    return vector / magnitude(vector)

"""Returns dot product of 2 vectors"""
def dot_product(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.dot(vector1, vector2)

"""Adds 2 vectors"""
def add_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    return vector1 + vector2

"""Scalar multiplication of a vector"""
def scalar_multiply(vector: np.ndarray, scalar: float) -> np.ndarray:
    return vector * scalar