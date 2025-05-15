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
W = np.zeros_like(positive_freqs)
positive_magnitude = magnitude[:n//2 + 1]

# Create weighting parameters
# ---- Parameters ----
scale = 1.0         # Global vertical scale (e.g. 1.0 for default, 0.5 to halve everything)
f_low = 0.5         # Lower edge of flat 0.4 region
f_mid_start = 2.0   # Start of ramp up (segment 2)
f_mid_end = 5.0     # End of ramp up (segment 2)
f_flat_start = f_mid_end  # Start of flat 1.0 region (segment 3)
f_flat_end = 16.0         # End of flat 1.0 region

# Value at the end of Segment 1 (flat) and start of Segment 2 (ramp)
val_low = 0.4 * scale
val_flat = 1.0 * scale

# Segment 0: 0.0–f_low — linear ramp to val_low
mask_0 = (positive_freqs > 0.0) & (positive_freqs < f_low)
W[mask_0] = val_low * (positive_freqs[mask_0] / f_low)

# Segment 1: f_low–f_mid_start — constant at val_low
mask_1 = (positive_freqs >= f_low) & (positive_freqs <= f_mid_start)
W[mask_1] = val_low

# Segment 2: f_mid_start–f_mid_end — linear ramp to val_flat
mask_2 = (positive_freqs > f_mid_start) & (positive_freqs <= f_mid_end)
W[mask_2] = val_low + (val_flat - val_low) * (
    (positive_freqs[mask_2] - f_mid_start) / (f_mid_end - f_mid_start)
)

# Segment 3: f_flat_start–f_flat_end — constant at val_flat
mask_3 = (positive_freqs > f_flat_start) & (positive_freqs <= f_flat_end)
W[mask_3] = val_flat

# Segment 4: > f_flat_end — decaying 6 dB/oct slope (W = val_flat * f_flat_end / f)
mask_4 = (positive_freqs > f_flat_end)
W[mask_4] = val_flat * f_flat_end / positive_freqs[mask_4]

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
# plt.loglog(positive_freqs, W) # Plot in m/s²
plt.semilogx(positive_freqs, 20 * np.log10(W)) # Plot in dB
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

# Vibration Serviceability Metrics 
rms_unweighted = np.sqrt(np.mean(acceleration ** 2))
rms_weighted = np.sqrt(np.mean(weighted_signal ** 2))
print(f"Unweighted RMS: {rms_unweighted:.6f} m/s^2")
print(f"Weighted RMS: {rms_weighted:.6f} m/s^2")

# Define your window parameters
window_duration = 1.0  # seconds
window_size = int(fs * window_duration)  # number of samples in 1 second

# Function to calculate running RMS
def running_rms(signal, window_size):
    return np.sqrt(np.convolve(signal**2, np.ones(window_size)/window_size, mode='valid'))

# Calculate running RMS for unweighted and weighted signals
rms_running_unweighted = running_rms(acceleration, window_size)
rms_running_weighted = running_rms(weighted_signal, window_size)

# Find the maximum RMS in those running windows
max_rms_unweighted = np.max(rms_running_unweighted)
max_rms_weighted = np.max(rms_running_weighted)

# Print results
print(f"MTVV 1s Running RMS (Unweighted): {max_rms_unweighted:.6f} m/s^2")
print(f"MTVV 1s Running RMS (Weighted):   {max_rms_weighted:.6f} m/s^2")

plt.show() 