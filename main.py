import numpy as np
from src.matrices import eigenvalues
from src.vectors import magnitude, normalise
from src.quantum import normalise_state, measurement_probabilities

v = np.array([3, 4])
m = np.array([[1, 2], [3, 4]])

print(magnitude(v))
print(normalise(v))

print(eigenvalues(m))

psi = np.array([[1],
                [1]], dtype=complex)

print(normalise_state(psi))
print(measurement_probabilities(psi))