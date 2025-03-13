import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import filtfilt, tf2zpk, bilinear, freqz



# Load accelerometer data from CSV (Ensure the CSV has 'Time' and 'Acceleration' columns)
data = pd.read_excel("siganaltestdata.xlsx")
time = data["Time"].values  # Time in seconds
acceleration = data["Acceleration"].values  # Acceleration in m/s²

# Compute sampling rate
dt = np.mean(np.diff(time))  # Time step (assuming uniform sampling)
fs = 1 / dt  # Sampling frequency (Hz)
N = len(acceleration)  # Number of data points

# Compute FFT
accel_fft = fft(acceleration)
freqs = fftfreq(N, d=dt)  # Frequency values

# Keep only the positive half of frequencies (real spectrum)
half_N = N // 2
freqs = freqs[:half_N]
magnitude = np.abs(accel_fft[:half_N]) / N  # Normalize FFT magnitude

# Plot the frequency spectrum
plt.figure(1)
plt.plot(freqs, magnitude, label="FFT Magnitude", color="r")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Accelerometer Data")
plt.grid()
plt.legend()

# Save FFT results to CSV
fft_data = pd.DataFrame({"Frequency": freqs, "Magnitude": magnitude})
fft_data.to_csv("fft_accelerometer_data.csv", index=False)

# Define 4 transfer functions
num1, den1 = [1], [1, 10]  # Transfer Function 1
num2, den2 = [1], [1, 5]   # Transfer Function 2
num3, den3 = [1], [1, 2]   # Transfer Function 3
num4, den4 = [1], [1, 1]   # Transfer Function 4

# Combine the transfer functions by multiplying numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    combined_num = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4)
    combined_den = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    return combined_num, combined_den

# Get the combined transfer function
combined_num, combined_den = combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4)

# Convert to discrete-time using bilinear transformation
b_discrete, a_discrete = bilinear(combined_num, combined_den, fs)

# Apply the transfer function using filtfilt
filtered_acceleration = filtfilt(b_discrete, a_discrete, acceleration)

# Compute FFT of the filtered data
filtered_accel_fft = fft(filtered_acceleration)

# Prevent division by zero when normalizing FFT magnitude
max_fft_value = np.max(np.abs(filtered_accel_fft))
if max_fft_value == 0:
    max_fft_value = 1e-10  # Avoid division by zero

filtered_magnitude = np.abs(filtered_accel_fft[:half_N]) / max_fft_value


# Plot the frequency spectrum of the filtered data
plt.figure(2)
plt.plot(freqs, filtered_magnitude, label="Filtered FFT Magnitude", color="b")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Filtered Accelerometer Data")
plt.grid()
plt.legend()

# Save FFT results of filtered data to CSV
fft_filtered_data = pd.DataFrame({"Frequency": freqs, "Magnitude": filtered_magnitude})
fft_filtered_data.to_csv("fft_filtered_accelerometer_data.csv", index=False)

from scipy.signal import freqs
import matplotlib.pyplot as plt
import numpy as np

# Compute frequency response
w, h = freqs(combined_num, combined_den, worN=np.logspace(-1, 3, 1000))  # Log scale from 0.1 Hz to 1000 Hz

# Plot frequency response
plt.figure(3)
plt.semilogx(w, 20 * np.log10(abs(h)))  # Convert magnitude to dB
plt.title("Frequency Response of Combined Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.show()
