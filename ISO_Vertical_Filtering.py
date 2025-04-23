import numpy as np
import pandas as pd
import math
import control
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import filtfilt, tf2zpk, bilinear, freqs
# Load the CSV file
data = pd.read_csv("results_500mc_trial_matrix.csv")

# Extract acceleration data for one trial (excluding index column)
acceleration = data.iloc[0, 1:].values

# Assumed sampling rate (Hz) — adjust this based on actual setup
fs = 100  # Hz
n = len(acceleration)
t = np.arange(n) / fs  # Time axis

# FFT calculation
acc_fft = fft(acceleration)
frequencies = fftfreq(n, 1/fs)
magnitude = np.abs(acc_fft) / n  # Normalize

# Plotting
# fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# Acceleration over time
plt.figure(figsize=(10, 5))
plt.plot(t, acceleration)
plt.title("Unfiltered Acceleration Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration")
plt.grid(True)


# Frequency spectrum (positive frequencies only)
positive_freqs = frequencies[:n // 2]
positive_magnitude = magnitude[:n // 2]
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude)
plt.title("Unfiltered Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.grid(True)
plt.tight_layout()

###################################################### MAKING ISO FILTER ############################################

# --- Wk Transfer Function Parameters ---# Band-Limiting
f1, f2 = 0.4, 100       # Band-limiting
f3, f4 = 12.5, 12.5      # Accel.-velocity transition
Q4 = 0.63
f5, f6 = 2.37, 3.35      # Upward step
Q5, Q6 = 0.91, 0.91

# Convert to angular frequencies (rad/s)
omega1 = 2 * math.pi * f1
omega2 = 2 * math.pi * f2
omega3 = 2 * math.pi * f3
omega4 = 2 * math.pi * f4
omega5 = 2 * math.pi * f5
omega6 = 2 * math.pi * f6

# HIGH PASS
num_h = [1,0,0]
den_h = [1,math.sqrt(2)*omega1,omega1**2]

# LOW PASS
num_l = 1
den_l = [omega2**-2,math.sqrt(2)/omega2,1]

# ACCEL.-VELOCITY TRANSITION
num_t = [omega3**-1,1]
den_t = [omega4**-2,(Q4*omega4)**-1,1]

# UPWARD STEP
num_s = [1,omega5/Q5,omega5**2];
den_s = [1,omega6/Q6,omega6**2];

# Combine the transfer functions by multiplying their numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    num = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4)
    den = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    return num, den

num_wk, den_wk = combine_transfer_functions(num_h, den_h, num_l, den_l, num_t, den_t, num_s, den_s)

# Frequency range for plotting (Hz)
frequencies = np.logspace(-1, 2.5, 1000)
omega = 2 * np.pi * frequencies

# Frequency response in dB
_, H = freqs(num_wk, den_wk, worN=omega)
H_dB = 20 * np.log10(np.abs(H))

# --- Plot (Magnitude in dB) ---
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, H_dB, label="|H(f)| in dB", color='tab:blue')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Wk Transfer Function (Magnitude in dB)")
plt.tight_layout()

###################################################### FILTERING DATA ############################################

omega = 2 * np.pi * frequencies  # Convert to rad/s

# FFT of acceleration signal
Xf = fft(acceleration)

from scipy.interpolate import interp1d

# Evaluate transfer function H(jω) at FFT frequency points
fft_frequencies = fftfreq(n, 1/fs)  # FFT frequency points
positive_indices = fft_frequencies >= 0  # Only positive frequencies

# Interpolate H to match FFT frequency points
H_interp = interp1d(frequencies, H, kind='linear', fill_value="extrapolate")
H_at_fft = H_interp(fft_frequencies[positive_indices])

# Filtered spectrum: multiply original FFT by H
Yf = H_at_fft * Xf[positive_indices]
magnitude_filtered = np.abs(Yf) / n

# Use only positive frequencies for plotting
positive_freqs = fft_frequencies[positive_indices]
positive_magnitude = magnitude_filtered

# Plot: Frequency vs Filtered Acceleration Magnitude
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude)
plt.title("Filtered Frequency Spectrum (Freq vs Acceleration)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Filtered Acceleration Magnitude")
plt.grid(True)
plt.tight_layout()
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.show()