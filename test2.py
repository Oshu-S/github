import numpy as np
import pandas as pd
import math
import control
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import filtfilt, tf2zpk, bilinear

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
num_s = [1,omega5/Q5,omega5**2]
den_s = [1,omega6/Q6,omega6**2]

# Combine the transfer functions by multiplying their numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    # Multiply numerators
    num_wk = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4) 
    
    # Multiply denominators
    den_wk = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    
    return num_wk, den_wk

# Get the combined transfer function
num_wk, den_wk = combine_transfer_functions(num_h, den_h, num_l, den_l, num_t, den_t, num_s, den_s)




from scipy.signal import freqs
import matplotlib.pyplot as plt
import numpy as np
# Compute frequency response
w, h = freqs(num_wk, den_wk, worN=np.logspace(np.log10(0.016), np.log10(250), 1000))  # Log scale from 0.016 Hz to 250 Hz

# Convert w from rad/s to Hz
f_hz = w / (2 * np.pi)

# Custom tick marks in Hz
custom_ticks = np.array([0.016, 0.0315, 0.063, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 31.5, 63, 125, 250])

# Plot frequency response
plt.figure(figsize=(8, 5))
plt.semilogx(f_hz, 20 * np.log10(abs(h)))  # Convert magnitude to dB
plt.title("Frequency Response of Combined Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frequency weightings (dB)")
plt.grid()
plt.xticks(custom_ticks, labels=[str(tick) for tick in custom_ticks])  # Apply custom frequency scale

plt.show()
