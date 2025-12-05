import numpy as np

k1=100
k2=200
k3=300
k4=500
k5=400
k6=150

k=np.array([
    [k2+k5+k6,  -k2 ,           -k6,            0],
    [-k2 ,      k1+k3+k2,       -k3,          -k1],
    [-k6,       -k3,          k3+k4+k6,       -k4],
    [0,          -k1,           -k4,      (k1+k4)]
],dtype=float)

b = np.array([20,10,30,30],dtype=float)
x=np.linalg.solve(k,b)

print("displacement : ",x)

