import numpy as np
#B=np.array([[2,1,1,10],
#           [3,2,3,18],
#            [1,4,9,16]],
#            dtype=float)

A=np.array([[ 750, -200, -150,    0,20],
            [-200,  600, -300, -100,10],
            [-150, -300,  950, -500,30],
            [   0, -100, -500,  600,30]],dtype=float)

n=len(A)

for i in range(n):
    pivot=A[i][i]
    A[i]=A[i]/pivot
    for j in range (i+1,n):
        A[j]=A[j]-A[j][i]*A[i]

x=np.zeros(n)

for i in range (n-1,-1,-1):
    x[i]=A[i][n]-np.sum(A[i][i+1:n]*x[i+1:n])

print("\n---Gauss Elimination SOLUTIONS---")
print("a=",x[0])
print("b=",x[1])
print("c=",x[2])
print("d=",x[3])

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

print("\n\n---LU decomposition Method=----")
print("L:\n", L)
print("U:\n", U)
print("Solution x:\n", x)
