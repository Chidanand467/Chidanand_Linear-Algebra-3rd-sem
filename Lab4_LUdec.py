import numpy as np

def lu_decompose(A):
    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]   
            L[j, i] = factor
            U[j] = U[j] -factor * U[i]  
    return L, U

def solve_LU(A, b):
    L, U = lu_decompose(A)
    n = A.shape[0]

    y = np.linalg.solve(L,b)
    x = np.linalg.solve(U,y)

    return L,U,x

A = np.array([[1, -3, 1],
              [0,4, 2],
              [0, 0, -6]])

b = np.array([-5, 10, -18])

L,U,x=solve_LU(A,b)

print("L:\n", L)
print("U:\n", U)
print("Solution x:\n", x)

