import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import freqs
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft


# Load data
data = pd.read_csv("results_500mc_trial_matrix.csv")
acceleration = data.iloc[0, 1:].values  # Use trial 0

# 1. Setup time and sampling
fs = 100  # Hz
dt = 1 / fs
n = len(acceleration)
t = np.arange(n) * dt

# 2. Frequency vector for FFT
freq_vector = fftfreq(n, d=dt)
acc_fft = fft(acceleration)
magnitude = np.abs(acc_fft) / n  # Normalize


# 3. Create weighting vector (same length as FFT)
weights = np.ones_like(freq_vector)

# Apply weighting only for positive frequencies (mirror for negative)
positive_freqs = freq_vector[:n//2 + 1]
W = np.ones_like(positive_freqs)
positive_magnitude = magnitude[:n//2 + 1]

# Apply weighting parameters
# Segment 0: 0.0–0.5 Hz — linear ramp up to 0.4 (matches 6 dB/octave)
mask_0 = (positive_freqs > 0.0) & (positive_freqs < 0.5)
W[mask_0] = 0.4 * positive_freqs[mask_0] / 0.5

# Segment 1: 0.5–2.0 Hz — constant 0.4
W[(positive_freqs >= 0.5) & (positive_freqs <= 2.0)] = 0.4

# Segment 2: 2.0–5.0 Hz — W(f) = f / 5
mask_2 = (positive_freqs > 2.0) & (positive_freqs <= 5.0)
W[mask_2] = positive_freqs[mask_2] / 5

# Segment 3: 5.0–16.0 Hz — constant 1.0
W[(positive_freqs > 5.0) & (positive_freqs <= 16.0)] = 1.0

# Segment 4: 16.0–100.0 Hz — W(f) = 16 / f
mask_4 = (positive_freqs > 16.0)
safe_freqs = np.copy(positive_freqs) # Guard against divide-by-zero in segment 4
safe_freqs[safe_freqs < 0.01] = 0.01 # If any values are exactly zero or very close to zero, 16.0 / f can cause instability.
W[mask_4] = 16.0 / safe_freqs[mask_4]

# Apply to full spectrum (positive + symmetric negative side)
weights[:n//2 + 1] = W

# Assign weights (mirror)
weights[:n//2 + 1] = W
if n % 2 == 0:
    weights[n//2 + 1:] = W[1:n//2][::-1]
else:
    weights[n//2 + 1:] = W[1:(n//2)+1][::-1]

# 4. Apply weighting to FFT
weighted_fft = acc_fft * weights

# 5. Inverse FFT to get time-weighted signal
weighted_signal = np.real(ifft(weighted_fft))

# 6. Compute magnitude spectrum of weighted FFT
weighted_magnitude = np.abs(weighted_fft) / n  # Normalized magnitude

# 7. Extract positive frequencies only
weighted_magnitude_pos = weighted_magnitude[:n//2 + 1]

# Plot frequency weighting
plt.figure(figsize=(10, 5))
plt.loglog(positive_freqs, W) # Plot in m/s²
# plt.semilogx(positive_freqs, 20 * np.log10(W)) # Plot in dB
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frequency Weighting (m/s²)")
plt.title("Asymptotic Approximation of Vertical Weighting")
plt.grid(True)
plt.tight_layout()

# Plotting Unfiltered Acceleration Time-History
plt.figure(figsize=(10, 5))
plt.plot(t, acceleration)
plt.title("Unfiltered Acceleration Time-History")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)

# Plotting Unweighted frequency spectrum (positive frequencies only)
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude)
plt.title("Unweighted Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Unweighted Acceleration (m/s²)")
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.grid(True)
plt.tight_layout()

# Plotting Weighted Frequency Spectrum
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, weighted_magnitude_pos, color='orange')
plt.title("Weighted Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Weighted Acceleration (m/s²)")
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.grid(True)
plt.tight_layout()

# compare both unweighted and weighted spectra on the same plot:
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude, label="Unweighted", color='blue')
plt.plot(positive_freqs, weighted_magnitude_pos, label="Weighted", color='orange')
plt.title("Comparison of Frequency Spectra")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 0.031)
plt.grid(True)
plt.tight_layout()

plt.show()