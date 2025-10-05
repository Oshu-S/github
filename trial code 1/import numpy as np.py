import numpy as np
from sympy import *

#mass = [1 for i in range(10)]
#M = np.diag(mass)

# Reshape M1 to be a 2D column vector
#M1 = (10 * np.array(mass)).reshape(-1, 1)

# Now stack them horizontally
#MM = np.hstack((M, M1))

#mp=np.array([1,2,3,4])
#phi_x = np.array([1,2,1,3])
#result_matrix = np.array([-mp[i] * phi_x for i in range(len(mp))])

#print(result_matrix)
import numpy as np
import random

# Define time array
t = np.array([1, 2, 3, 4, 5])
psd_force = 0
# Loop to calculate and print forces
for i in range(5):
    phi = random.uniform(0, 2 * np.pi)  # Random phase
    force = np.sqrt(4) * np.cos(np.pi / 2 * t)  # Calculate force at given times
    print("force",force)
    psd_force += force
    print("psd_force",psd_force)

print(psd_force)