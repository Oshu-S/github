import numpy as np
import pandas as pd
import math
import control
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import filtfilt, tf2zpk, bilinear

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
plt.ylabel("Acceleration ($\mathregular{ms^{-2}}$)")
plt.title("Frequency Spectrum of Accelerometer Data")
plt.grid()
plt.legend()

# Save FFT results to CSV
fft_data = pd.DataFrame({"Frequency": freqs, "Magnitude": magnitude})
fft_data.to_csv("fft_accelerometer_data.csv", index=False)


## FILTERING ##

# Wk
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
num_s = [1,omega5/Q5,omega5**2];
den_s = [1,omega6/Q6,omega6**2];

# Combine the transfer functions by multiplying their numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    # Multiply numerators
    num_wk = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4) 
    
    # Multiply denominators
    den_wk = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    
    return num_wk, den_wk

# Get the combined transfer function
num_wk, den_wk = combine_transfer_functions(num_h, den_h, num_l, den_l, num_t, den_t, num_s, den_s)

# Print transfer function for Wk
H = control.tf(num_wk, den_wk)
print(H)

# Convert to discrete-time using bilinear transformation
b_discrete, a_discrete = bilinear(num_wk, den_wk, fs)

# Apply the transfer function using filtfilt
filtered_acceleration = filtfilt(b_discrete, a_discrete, acceleration)

# Compute FFT of the filtered data
filtered_accel_fft = fft(filtered_acceleration)

# # Prevent division by zero when normalizing FFT magnitude
# max_fft_value = np.max(np.abs(filtered_accel_fft))
# if max_fft_value == 0:
#     max_fft_value = 1e-10  # Avoid division by zero
#filtered_magnitude = np.abs(filtered_accel_fft[:half_N]) / max_fft_value
filtered_magnitude = np.abs(filtered_accel_fft[:half_N]) / N  # Normalize FFT magnitude

# Plot the frequency spectrum of the filtered data
plt.figure(2)
plt.plot(freqs, filtered_magnitude, label="Filtered FFT Magnitude", color="b")
# plt.semilogx(freqz, 20 * np.log10(abs(filtered_magnitude)))  # Convert magnitude to dB
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration ($\mathregular{ms^{-2}}$)")
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
w, h = freqs(num_wk, den_wk, worN=np.logspace(-1, 3, 1000))  # Log scale from 0.1 Hz to 1000 Hz
log_ticks = 2 ** np.linspace(np.log2(min(w)), np.log2(max(w)), num=10) # For base-2
# Plot frequency response
plt.figure(3)
plt.semilogx(w, 20 * np.log10(abs(h)))  # Convert magnitude to dB
plt.title("Frequency Response of Combined Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Frequency weightings (dB)")
plt.grid()
plt.xticks(log_ticks, [f'{tick:.1f}' for tick in log_ticks]) # Customize the ticks on x-axis for non-logarithmic powers
plt.axis([0.05, 500, -90, 10])
plt.show()

