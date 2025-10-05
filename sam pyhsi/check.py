import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

# Generate an example signal: combination of two sine waves and noise
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)  # Time vector for 1 second
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))

# Perform Empirical Mode Decomposition (EMD)
emd = EMD()
IMFs = emd.emd(signal)

# Plot the original signal and the IMFs
plt.figure(figsize=(10, 8))

plt.subplot(len(IMFs) + 1, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')

for i, IMF in enumerate(IMFs):
    plt.subplot(len(IMFs) + 1, 1, i + 2)
    plt.plot(t, IMF)
    plt.title(f'IMF {i+1}')

plt.tight_layout()
plt.show()

from scipy.signal import welch

# Compute and plot the PSD of each IMF
fs = 1000  # Sampling frequency

for i, IMF in enumerate(IMFs):
    frequencies, psd_values = welch(IMF, fs=fs, nperseg=256)
    
    plt.figure()
    plt.semilogy(frequencies, psd_values)
    plt.title(f'PSD of IMF {i+1}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.show()
