#Coefficient Matrrix
import numpy as np
A=np.array([[0,(4/5),(4/5)],
            [-1,0,-(3/5)], 
            [1,(3/5),0]])

#vector b
b=np.array([600,0,0])

#Solution
S=np.linalg.solve(A,b)
print("F1=",S[0] )
print("F3=",S[1] )
print("F2=",S[2] )

