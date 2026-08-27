import numpy as np

"""Returns matrix multiplication of 2 matrices"""
def matrix_multiply(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    return matrix1 @ matrix2

"""Returns transpose of a matrix"""
def transpose(matrix: np.ndarray) -> np.ndarray:
    return matrix.T

"""Returns determinant of a matrix"""
def determinant(matrix: np.ndarray) -> float:
    return np.linalg.det(matrix)

"""Returns inverse of a matrix"""
def inverse(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)

"""Returns scalar multiplication of a matrix"""
def scalar_multiply(scalar: float, matrix: np.ndarray) -> np.ndarray:
    return scalar * matrix

"""Returns addition of 2 matrices"""
def add_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    return matrix1 + matrix2

"""Returns eigenvalues of a matrix"""
def eigenvalues(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eig(matrix)[0]

"""Returns eigenvectors of a matrix"""
def eigenvectors(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eig(matrix)[1]

"""Returns whether a matrix is hermitian"""
def is_hermitian(matrix):
    return np.allclose(matrix, matrix.conj().T)

"""Returns whether a matrix is unitary"""
def is_unitary(matrix):
    return np.allclose(matrix @ matrix.conj().T, np.eye(matrix.shape[0]))