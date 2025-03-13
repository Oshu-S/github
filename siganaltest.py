import numpy as np
import pandas as pd
import math
import control
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import lfilter

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
plt.figure(figsize=(8, 5))
plt.plot(freqs, magnitude, label="FFT Magnitude", color="r")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Accelerometer Data")
plt.grid()
plt.legend()
plt.show()

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
num_s = [omega5**-2,(Q5*omega5)**-1,1]
den_s = [omega6**-2,(Q6*omega6)**-1,1]

# Combine the transfer functions by multiplying their numerators and denominators
def combine_transfer_functions(num1, den1, num2, den2, num3, den3, num4, den4):
    # Multiply numerators
    # num_wk = np.polymul(np.polymul(np.polymul(num1, num2), num3), num4) ###############################################################################################
    
    # Multiply denominators
    den_wk = np.polymul(np.polymul(np.polymul(den1, den2), den3), den4)
    
    return num_wk, den_wk

# Get the combined transfer function
num_wk = [2.874e-05, 0.002727, 0.04331, 0.5005, 0, 0] ###################################################################################################################
num_wk, den_wk = combine_transfer_functions(num_h, den_h, num_l, den_l, num_t, den_t, num_s, den_s)

# from scipy.signal import freqs
# import matplotlib.pyplot as plt
# import numpy as np

# # Compute frequency response
# w, h = freqs(num_wk, den_wk, worN=np.logspace(-1, 3, 1000))  # Log scale from 0.1 Hz to 1000 Hz

# # Plot frequency response
# plt.figure()
# plt.semilogx(w, 20 * np.log10(abs(h)))  # Convert magnitude to dB
# plt.title("Frequency Response of Combined Filter")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude (dB)")
# plt.grid()
# plt.show()

H = control.tf(num_wk, den_wk)
poles = control.pole(H)
print("Poles:", poles)


# Apply the transfer function to the data using lfilter
def apply_transfer_function(data, num, den):
    filtered_data = lfilter(num, den, data)
    return filtered_data

# Apply the transfer function to the accelerometer data
filtered_acceleration = apply_transfer_function(acceleration, num_wk, den_wk)

print(num_wk)


# Compute FFT of the filtered data
filtered_accel_fft = fft(filtered_acceleration)
filtered_magnitude = np.abs(filtered_accel_fft[:half_N]) / N  # Normalize FFT magnitude

# Plot the frequency spectrum of the filtered data
plt.figure(figsize=(8, 5))
plt.plot(freqs, filtered_magnitude, label="Filtered FFT Magnitude", color="b")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Filtered Accelerometer Data")
plt.grid()
plt.legend()
plt.show()

# Save FFT results of filtered data to CSV
fft_filtered_data = pd.DataFrame({"Frequency": freqs, "Magnitude": filtered_magnitude})
fft_filtered_data.to_csv("fft_filtered_accelerometer_data.csv", index=False)

print("FFT Min Magnitude:", np.min(filtered_magnitude))
print("FFT Max Magnitude:", np.max(filtered_magnitude))
print("FFT Contains NaN:", np.any(np.isnan(filtered_magnitude)))
print("FFT Contains Inf:", np.any(np.isinf(filtered_magnitude)))

