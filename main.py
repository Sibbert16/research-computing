import numpy as np
from src.matrices import eigenvalues
from src.vectors import magnitude, normalise

v = np.array([3, 4])
m = np.array([[1, 2], [3, 4]])

print(magnitude(v))
print(normalise(v))

print(eigenvalues(m))