import numpy as np
#B=np.array([[2,1,1,10],
#           [3,2,3,18],
#            [1,4,9,16]],
#            dtype=float)

A=np.array([[ 750, -200, -150,    0,20],[-200,  600, -300, -100,10],[-150, -300,  950, -500,30],[   0, -100, -500,  600,30]],dtype=float)

n=len(A)

for i in range(n):
    pivot=A[i][i]
    A[i]=A[i]/pivot
    for j in range (i+1,n):
        A[j]=A[j]-A[j][i]*A[i]

x=np.zeros(n)

for i in range (n-1,-1,-1):
    x[i]=A[i][n]-np.sum(A[i][i+1:n]*x[i+1:n])

print("SOLUTIONS:")
print("x=",x[0])
print("y=",x[1])
print("z=",x[2])
print("a=",x[3])