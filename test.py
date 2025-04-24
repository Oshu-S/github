import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import freqs
from scipy.interpolate import interp1d

# Load data
data = pd.read_csv("results_500mc_trial_matrix.csv")
acceleration = data.iloc[0, 1:].values  # Use trial 0

# Time domain setup
fs = 100  # Hz (time step = 0.01 s)
dt = 1 / fs
n = len(acceleration)
t = np.arange(n) * dt

# ----------------- FFT (Unfiltered Signal) -----------------
acc_fft = fft(acceleration)
frequencies = fftfreq(n, dt)
magnitude = np.abs(acc_fft) / n  # Normalize

# Keep only positive frequencies for plotting
positive_freqs = frequencies[:n // 2]
positive_magnitude = magnitude[:n // 2]

# Plotting Unfiltered Acceleration Time-History
plt.figure(figsize=(10, 5))
plt.plot(t, acceleration)
plt.title("Unfiltered Acceleration Time-History")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)

# Frequency spectrum (positive frequencies only)
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude)
plt.title("Unfiltered Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (m/s²)")
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.grid(True)
plt.tight_layout()

# ----------------- BS 6841 Filter - Wb, Vertical -----------------
f1, f2 = 0.4, 100       # Band-limiting
Q1 = 0.71
f3, f4 = 16, 16     # Frequency Weighting
f5 = 2.5
f6 = 4
Q2 = 0.55
Q3 = 0.9
Q4 = 0.95
K = 0.4

# Band-limiting
num_b = [4*math.pi**2*f2**2,0,0]
den_b = [1, (2*math.pi*f1/Q1) + (2*math.pi*f2/Q1), (4*f1**2*math.pi**2) + (4*f1*f2*math.pi**2)/(Q1**2) + (4*f2**2*math.pi**2), (8*f1**2*f2*math.pi**3/Q1) + (8*f1*f2**2*math.pi**3/Q1), 16*f1**2*f2**2*math.pi**4]
# # S**2
# (4*f1**2*math.pi**2) + (4*f1*f2*math.pi**2)/(Q1**2) + (4*f2**2*math.pi**2)
# # S
# (8*f1**2*f2*math.pi**3/Q1) + (8*f1*f2**2*math.pi**3/Q1)
# # C
# 16*f1**2*f2**2*math.pi**4

# Frequency Weighting
num_w = 

# Combine the transfer functions by multiplying their numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    num = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4)
    den = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    return num, den

num_wb, den_wb = combine_transfer_functions(num_h, den_h, num_l, den_l, num_t, den_t, num_s, den_s)

# Plotting BS 6841 Filter on log-log scale
freq_range = np.logspace(-1, 2.5, 1000)
omega = 2 * np.pi * freq_range
_, H = freqs(num_wb, den_wb, worN=omega)
H_dB = 20 * np.log10(np.abs(H)) # Convert to dB


plt.figure(figsize=(10, 5))
plt.semilogx(freq_range, H_dB) # ISO 2631 & AS 2670 Axes

# BS 6841 & BS 6472 Axes
# plt.loglog(freq_range, np.abs(H))
# plt.xlim(0.01, 100)
# plt.ylim(0.01, 10)

plt.grid(True, which="both", ls="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("BS 6841 - Wb Filter")
plt.tight_layout()

# ----------------- Filtering -----------------
# Interpolate H onto FFT frequency bins
H_interp = interp1d(freq_range, H, kind='linear', fill_value="extrapolate")
H_vals = H_interp(np.abs(frequencies ))  # Use abs to get symmetric frequency behavior
Yf = acc_fft * H_vals  # Apply filter --> Multiply acceleration by Tranfer Function

# Get magnitude of filtered result
filtered_magnitude = np.abs(Yf[:n//2]) / n
filtered_freqs = frequencies [:n//2]

# Plot: Frequency vs Filtered Acceleration Magnitude
plt.figure(figsize=(10, 5))
plt.plot(filtered_freqs, filtered_magnitude)
plt.title("Filtered Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.tight_layout()
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.show()