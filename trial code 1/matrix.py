import math
import numpy as np
from sympy import *
"""
Bridge matrices definition
"""
class bridge:
    """
    Class definition
    """
    def __init__(self, length, modulus, density, damp, numbers):
        self.L = length
        self.EI = modulus
        self.rho = density
        self.damp = damp
        self.n = numbers

    """
    Construct a simply supported beam by superposition method
    Parameters
    ----------
    length : Span length
        Unit m.
    modulous : flexural rigidities
        Unit Nm^2.
    density : unit density
        Unit kg/m.
    damp : damping percentage
        example 0.003 means 0.3%.
    numbers : number of mode shapes considered
        first 10 modes could be sufficient.
    Returns
    -------
    None.
    """
    def Mass_matrix(self):
        """
        Returns the mass matrix which is normalized for the bridge.
        -------
        None.
        """
        mass = [1 for i in range(self.n)]
        mass = np.diag(mass)
        return mass

    def Stiffness_matrix(self):
        """
        Returns normalized stiffness matrix of bridge.
        -------
        None.
        """
        k = []
        for i in range(self.n):
            it = i + 1
            k.append(self.EI / self.rho * (it * np.pi / self.L) ** 4)
        k = np.diag(k)
        return k
    
    def Damp_matrix(self):
        """
        Returns rayleigh damping matrix.
        -------
        None.
        """
        M = bridge.Mass_matrix(self)
        K = bridge.Stiffness_matrix(self)
        w1 = K[0, 0] ** 0.5
        w2 = K[1, 1] ** 0.5
        a1 = 2 * self.damp * w1 * w2 / (w1 + w2)
        a2 = 2 * self.damp / (w1 + w2)
        c = a1 * M + a2 * K
        return c 
    
 