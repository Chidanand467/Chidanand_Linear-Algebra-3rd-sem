import numpy as np

v1 = np.array([1,0,2,1,0])
v2 = np.array([0,1,1,3,2])
v3 = np.array([2,1,1,4,1])
v4 = np.array([1,2,1,1,2])
v5 = np.array([3,1,4,2,3])

A = np.column_stack([v1,v2,v3,v4,v5])

dimension = np.linalg.matrix_rank(A)

print("Dimension of span:", dimension)
