import pickle

# Replace 'your_file.pkl' with your actual file path
with open('results_500mc_trial_matrix.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)


# import numpy as np
# import pandas as pd
# import math
# import control
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft, fftfreq
# from scipy.signal import filtfilt, tf2zpk, bilinear

# Load accelerometer data from CSV (Ensure the CSV has 'Time' and 'Acceleration' columns)
# data = pd.read_pickle("results_500mc_trial_matrix.pkl")
# print(data.iloc[0, 3])

# time = data["Time"].values  # Time in seconds
# acceleration = data["Acceleration"].values  # Acceleration in m/s²

# # Compute sampling rate
# dt = np.mean(np.diff(time))  # Time step (assuming uniform sampling)
# fs = 1 / dt  # Sampling frequency (Hz)
# N = len(acceleration)  # Number of data points

# # Compute FFT
# accel_fft = fft(acceleration)
# freqs = fftfreq(N, d=dt)  # Frequency values

# # Keep only the positive half of frequencies (real spectrum)
# half_N = N // 2
# freqs = freqs[:half_N]
# magnitude = np.abs(accel_fft[:half_N]) / N  # Normalize FFT magnitude

# # Plot the frequency spectrum
# plt.figure(1)
# plt.plot(freqs, magnitude, label="FFT Magnitude", color="r")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Acceleration ($\mathregular{ms^{-2}}$)")
# plt.title("Frequency Spectrum of Accelerometer Data")
# plt.grid()
# plt.legend()

# # Save FFT results to CSV
# fft_data = pd.DataFrame({"Frequency": freqs, "Magnitude": magnitude})
# fft_data.to_csv("fft_accelerometer_data.csv", index=False)
