import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import lfilter

# Band-Limiting
f1 = 0.4 # Hz
f2 = 100 # Hz
# Acceleration-velocity transition
f3 = 12.5 # Hz
f4 = 12.5 # Hz
Q4 = 0.63
# Upward step
f5 = 2.37 # Hz
Q5 = 0.91
f6 = 3.35 # Hz
Q6 = 0.91

# HIGH PASS
omega1 = 2*math.pi*f1
num_h = [1,0,0]
den_h = [1,math.sqrt(2)*omega1,omega1**2]

# LOW PASS
omega2 = 2*math.pi*f2
num_l = 1
den_l = [omega2**-2,math.sqrt(2)/omega2,1]

# ACCEL.-VELOCITY TRANSITION
omega3 = 2*math.pi*f3
omega4 = 2*math.pi*f4
num_t = [omega3**-1,1]
den_t = [omega4**-2,(Q4*omega4)**-1,1]

# UPWARD STEP
omega5 = 2*math.pi*f5
omega6 = 2*math.pi*f6
num_s = [omega5**-2,(Q5*omega5)**-1,1]
den_s = [omega6**-2,(Q6*omega6)**-1,1]

# Combine the transfer functions by multiplying their numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    # Multiply numerators
    num_wk = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4)
    
    # Multiply denominators
    den_wk = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    
    return num_wk, den_wk

# Get the combined transfer function
num_wk, den_wk = combine_transfer_functions(num_h, den_h, num_l, den_l, num_t, den_t, num_s, den_s)

print(num_wk)
print(den_wk)