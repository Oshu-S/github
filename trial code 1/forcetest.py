from matrix import*  
from solver import* 
from pedestrian import* 
from matplotlib import pyplot as plt
import timeit
import numpy as np
from test import Human

def indicat(x, lb, numped):
    """"
    Indicator function which checks whether the pedestrian is or not on the bridge, it
    returns 1 if it is on the bridge, otherwise zero.
    Parameters
    ----------
    x : vector shows the positions of wheels
        Unit m.
    lb : single span length
        Unit m.
    numped : number of pedestrians
       
        Returns
    -------
    None
    """
    I = np.zeros((numped))  # important
    for i in range(numped):
        if 0 < x[i] < lb:
            I[i] = 1
        else:
            I[i] = 0
    return I

t=np.arange(0, (50+1) / 1.5, 0.05)
xrb=np.zeros(len(t))

for j in range(len(t)):
  xrb[j] = 1.5*t[j]
print(xrb)

#n = 3
#N = np.zeros((n, len(xrb)))
#for i in range(n):
        #for j in range(len(xrb)):
         # it = i + 1
          #N[i][j] = (2 / (500*50)) ** 0.5 * sin(it * pi * xrb[j] / 50) #x interest should be time dependednt x=vt xrb is a matrix with time varying location of each pedestrian
#I = indicat(xrb, 50, 1) 
#NN = np.dot(N, np.diag(I))

#print(N)

F = calcPedForce(Human,t)
print(F)